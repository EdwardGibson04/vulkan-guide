---
layout: default
title: Improving the render loop
parent:  "2. Drawing with Compute"
nav_order: 1
---

Before we begin drawing, we need to implement a couple other things. First we have a deletion queue that will allow us to safely handle the cleanup of a growing amount of objects, and then we will change the render loop to draw into a non-swapchain image and then copy it to the swapchain.


## Deletion queue
As we begin to add more and more vulkan structures, we need a way to handle their destruction. We could keep adding more things into the `cleanup()` function, but that would not scale and would be very annoying to keep synced correctly. 
We are going to add a new structure to the engine, called a DeletionQueue. This is a common approach by lots of engines, where we add the objects we want to delete into some queue, and then run that queue to delete all the objects in the correct orders.
In our implementation, we are going to keep it simple, and have it store std::function callbacks in a deque. We will be using that deque as a First In Last Out queue, so that when we flush the deletion queue, it first destroys the objects that were added into it last.

This is the entire implementation.

```cpp
struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		deletors.clear();
	}
};
```

std::function stores a lambda, and we can use it to store a callback with some data, which is perfect for this case. 

Doing callbacks like this is inneficient at scale, because we are storing whole std::functions for every object we are deleting, which is not going to be optimal. For the amount of objects we will use in this tutorial, its going to be fine. but if you need to delete thousands of objects and want them deleted faster, a better implementation would be to store arrays of vulkan handles of various types such as VkImage, VkBuffer, and so on. And then delete those from a loop.

We will have the deletion queue in multiple places, for multiple lifetimes of objects. One of them is on the engine class itself, and will be flushed when the engine gets destroyed. Global objects go into that one. We will also store one deletion queue for each frame in flight, which will allow us to delete objects next frame after they are used. 

Add it into VulkanEngine class inside the main class, and inside the FrameData struct

```cpp
struct FrameData {
	 //other data
      DeletionQueue _deletionQueue;
};

class VulkanEngine{
    //other data
    DeletionQueue _mainDeletionQueue;
}

```

We then call it from 2 places, right after we wait on the Fence per frame, and from the cleanup() function after the WaitIdle call. By flushing it right after the fence, we make sure that the GPU has finished executing that frame so we can safely delete objects create for that specific frame only. We also want to make sure we free those per-frame resources when destroying the rest of frame data.

```cpp
void VulkanEngine::draw()
{
	//wait until the gpu has finished rendering the last frame. Timeout of 1 second
	VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));

	get_current_frame()._deletionQueue.flush();

    //other code
}

void VulkanEngine::cleanup()
{	
	if (_isInitialized) {
		
		//make sure the gpu has stopped doing its things				
		vkDeviceWaitIdle(_device);
		
		//free per-frame structures and deletion queue
		for (int i = 0; i < FRAME_OVERLAP; i++) {

			vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

			//destroy sync objects
			vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
			vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
			vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

			_frames[i]._deletionQueue.flush();
		}

		//flush the global deletion queue
		_mainDeletionQueue.flush();

		//rest of cleanup function
	}
}
```

With the deletion queue set, now whenever we create new vulkan objects we can just add them into the queue.

## Memory Allocation

To improve the render loop, we will need to allocate a image, and this gets us into how to allocate objects in vulkan. We are going to skip that entire chapter, because we will be using Vulkan Memory Allocator library. Dealing with the different memory heaps and object restrictions such as image alignment is very error prone and really hard to get right, specially if you want to get it right at a decent performance. By using VMA, we skip all that, and we get a battle tested way that is guaranteed to work well. There are cases like the PCSX3 emulator project, where they replaced their attempt at allocation to VMA, and won 20% extra framerate. 

vk_types.h already holds the include needed for the VMA library, but we need to do something else too.

On vk_engine.cpp we include it too, but with `VMA_IMPLEMENTATION` defined.

```cpp
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
```

VMA holds both the normal header, and the implementation of the functions into the same header file. We need to define `VMA_IMPLEMENTATION` exactly into only one of the .cpp files of our project, and that will store and compile the definitions for the VMA functions. 

Add the allocator to the VulkanEngine class
```cpp
class VulkanEngine{

    VmaAllocator _allocator;
}
```

Now we will initialize it from `init_vulkan()` call, at the end of the function.

^code vma_init chapter-2/vk_engine.cpp

There isnt much to explain it, we are initializing the _allocator member, and then adding its destruction function into the destruction queue so that it gets cleared when the engine exits. We hook the physical device, instance, and device to the creation function. We give the flag `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT` that will let us use GPU pointers later when we need them.
Vulkan Memory Allocator library follows similar call conventions as the vulkan api, so it works with similar info structs.

# New draw loop.

Drawing directly into the swapchain is fine for many projects, and it can even be optimal in some cases such as phones. But it comes with a few restrictions. 
The most important of them is that the formats of the image used in the swapchain are not guaranteed. Different OS, drivers, and windowing modes can have different optimal swapchain formats. Things like HDR support also need their own very specific formats.
Another one is that we only get a swapchain image index from the windowing present system. There are low-latency techniques where we could be rendering into another image, and then directly push that image to the swapchain with very low latency. 

One very important limitation is that their resolution is fixed to whatever your window size is. If you want to have higher or lower resolution, and then do some scaling logic, you need to draw into a different image.

And last, swapchain formats are, for the most part, low precision. Some platforms with High Dynamic Range rendering have higher precision formats, but you will often default to 8 bits per color. So if you want high precision light calculations, system that would prevent banding, or to be able to go past 1.0 on the normalized color range, you will need a separate image for drawing.

For all those reasons, we will do the whole tutorial rendering into a separate image than the one from the swapchain. After we are doing with the drawing, we will then copy that image into the swapchain image and present it to the screen.

The image we will be using is going to be in the  RGBA 16-bit float format. This is slightly overkill, but will provide us with a lot of extra precision that will come in handy when doing lighting calculations and better rendering.


# Vulkan Images
We have already dealt superficially with images when setting up the swapchain, but it was handled by VkBootstrap. This time we will create the images ourselves.


Lets begin by adding the new members we will need to the VulkanEngine class.

On vk_types.h, add this structure which holds the data needed for an image. We will hold a `VkImage` alongside its default `VkImageView`, then the allocation for the image memory, and last, the image size and its format, which will be useful when dealing with the image. We also add a `_drawExtent` that we can use to decide what size to render.

```cpp
struct AllocatedImage {
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};
```

```cpp
class VulkanEngine{

	//draw resources
	AllocatedImage _drawImage;
	VkExtent2D _drawExtent;
}
```

Lets check the vk_initializers function for image and imageview create info.

^code image_set shared/vk_initializers.cpp

We will hardcode the image tiling to OPTIMAL, which means that we allow the gpu to shuffle the data however it sees fit. If we want to read the image data from cpu, we would need to use tiling LINEAR, which makes the gpu data into a simple 2d array. This tiling highly limits what the gpu can do, so the only real use case for LINEAR is CPU readback.

On the imageview creation, we need to setup the subresource. Thats similar to the one we used in the pipeline barrier.

Now, at the end of init_swapchain, lets create it.

^code init_swap chapter-2/vk_engine.cpp

We begin by creating a VkExtent3d structure with the size of the image we want, which will match our window size. We copy it into the AllocatedImage

Then, we need to fill our usage flags. In vulkan, all images and buffers must fill a UsageFlags with what they will be used for. This allows the driver to perform optimizations in the background depending on what that buffer or image is going to do later. In our case, we want TransferSRC and TransferDST so that we can copy from and into the image,  Storage because thats the "compute shader can write to it" layout, and Color Attachment so that we can use graphics pipelines to draw geometry into it.

The format is going to be `VK_FORMAT_R16G16B16A16_SFLOAT`. This is 16 bit floats for all 4 channels, and will use 64 bits per pixel. Thats a fair amount of data, 2x what a 8 bit color image uses, but its going to be useful.

When creating the image itself, we need to send the image info and an alloc info to VMA. VMA will do the vulkan create calls for us and directly give us the vulkan image. 
The interesting thing in here is Usage and the required memory flags.
With VMA_MEMORY_USAGE_GPU_ONLY usage, we are letting VMA know that this is a gpu texture that wont ever be accessed from CPU, which lets it put it into gpu VRAM. To make extra sure of that, we are also setting `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` as a memory flag. This is a flag that only gpu-side VRAM has, and guarantees the fastest access.

In vulkan, there are multiple memory regions we can allocate images and buffers from. PC implementations with dedicated GPUs will generally have a cpu ram region, a GPU Vram region, and a "upload heap" which is a special region of gpu vram that allows cpu writes. If you have resizable bar enabled, the upload heap can be the entire gpu vram. Else it will be much smaller, generally only 256 megabytes. We tell VMA to put it on GPU_ONLY which will prioritize it to be on the gpu vram but outside of that upload heap region.

With the image allocated, we create an imageview to pair with it. In vulkan, you need a imageview to access images. This is generally a thin wrapper over the image itself that lets you do things like limit access to only 1 mipmap. We will always be pairing vkimages with their "default" imageview in this tutorial.

# New draw loop

Now that we have a new draw image, lets add it into the render loop.

We will need a way to copy images, so add this into vk_images.cpp

^code copyimg shared/vk_images.cpp
 
Also add the corresponding declaration to `vk_images.h`
```cpp
	namespace vkutil {

	    // Existing:
	    void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

		void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize);	
	}
```

Vulkan has 2 main ways of copying one image to another. you can use VkCmdCopyImage or VkCmdBlitImage.
CopyImage is faster, but its much more restricted, for example the resolution on both images must match.
Meanwhile, blit image lets you copy images of different formats and different sizes into one another. You have a source rectangle and a target rectangle, and the system copies it into its position. Those two functions are useful when setting up the engine, but later its best to ignore them and write your own version that can do extra logic on a fullscreen fragment shader.

With it, we can now update the render loop. As draw() is getting too big, we are going to leave the syncronization, command buffer management, and transitions in the draw() function, but we are going to add the draw commands themselves into a draw_background() function.

```cpp
void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
^code draw_clear chapter-2/vk_engine.cpp
}
```

Add the function to the header too.

We will be changing the code that records the command buffer. You can now delete the older one. 
The new code is this.
```cpp
^code draw_first chapter-2/vk_engine.cpp

	// execute a copy from the draw image into the swapchain
	vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

	// set swapchain image layout to Present so we can show it on the screen
	vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	//finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(cmd));
```

The main difference we have in the render loop is that we no longer do the clear on the swapchain image. Instead, we do it on the `_drawImage.image`. Once we have cleared the image, we transition both the swapchain and the draw image into their layouts for transfer, and we execute the copy command. Once we are done with the copy command, we transition the swapchain image into present layout for display. As we are always drawing on the same image, our draw_image does not need to access swapchain index, it just clears the draw image. We are also writing the _drawExtent that we will use for our draw region.

This will now provide us a way to render images outside of the swapchain itself. We now get significantly higher pixel precision, and we unlock some other techniques.

With that done, we can now move into the actual compute shader execution steps.

^nextlink

{% include comments.html term="Vkguide 2 Beta Comments" %}