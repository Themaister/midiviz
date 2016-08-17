#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "midi.hpp"
#include <sndfile.h>
#include <mutex>
#include <atomic>
#include <vector>

#include "vulkan/vulkan_symbol_wrapper.h"
#include <libretro_vulkan.h>

static struct retro_hw_render_callback hw_render;
static const struct retro_hw_render_interface_vulkan *vulkan;

static MIDI::File midi_file;
static SNDFILE *audio_file;
static bool use_audio_cb;
static std::mutex audio_lock;
static std::atomic_bool audio_cb_enable;
static std::atomic_uint audio_frames;

#define BASE_WIDTH 1280
#define BASE_HEIGHT 720
#define MAX_SYNC 8

#define NUM_PARTICLES (64 * 1024)
#define FRAMES (44100 / 60)

static unsigned width  = BASE_WIDTH;
static unsigned height = BASE_HEIGHT;

struct buffer
{
   VkBuffer buffer;
   VkDeviceMemory memory;
};

static void free_buffer(VkDevice device, buffer *buf)
{
   vkDestroyBuffer(device, buf->buffer, nullptr);
   vkFreeMemory(device, buf->memory, nullptr);
   memset(buf, 0, sizeof(*buf));
}

struct vulkan_data
{
   unsigned index;
   unsigned num_swapchain_images;
   uint32_t swapchain_mask;

   buffer vbo;
   buffer positions;
   buffer velocity;
   buffer color;
   unsigned particle_ptr;

   VkPhysicalDeviceMemoryProperties memory_properties;
   VkPhysicalDeviceProperties gpu_properties;

   VkDescriptorSetLayout set_layout;
   VkDescriptorPool desc_pool;
   VkDescriptorSet desc_set;

   VkPipelineCache pipeline_cache;
   VkPipelineLayout pipeline_layout;
   VkPipelineLayout compute_pipeline_layout;
   VkRenderPass render_pass;

   VkPipeline particle_pipeline;
   VkPipeline generate_pipeline;
   VkPipeline move_pipeline;
   VkPipeline pluck_pipeline;
   VkPipeline kick_pipeline;
   VkPipeline snare_pipeline;
   VkPipeline arp_pipeline;
   VkPipeline bass_pipeline;
   VkPipeline lead_pipeline;
   VkPipeline piano_pipeline;

   struct retro_vulkan_image images[MAX_SYNC];
   VkDeviceMemory image_memory[MAX_SYNC];
   VkFramebuffer framebuffers[MAX_SYNC];
   VkCommandPool cmd_pool[MAX_SYNC];
   VkCommandBuffer cmd[MAX_SYNC];
};
static struct vulkan_data vk;

struct particle_state
{
   float r = 0.0f;
   float g = 0.0f;
   float b = 0.0f;
   float kick = 0.0f;
   float snare = 0.0f;
   float kick_beat = 0.0f;
   float snare_beat = 0.0f;

   double frame;
   double last_lead[128];
   bool lead[128];
   float lead_velocity[128];
   double last_bass[128];
   bool bass[128];
   float bass_velocity[128];

   unsigned end_counter;
   bool left, right;
};
static particle_state state;

static void reset_state()
{
   memset(state.lead, 0, sizeof(state.lead));
   memset(state.last_lead, 0, sizeof(state.last_lead));
   memset(state.lead_velocity, 0, sizeof(state.lead_velocity));
   memset(state.bass, 0, sizeof(state.bass));
   memset(state.last_bass, 0, sizeof(state.last_bass));
   memset(state.bass_velocity, 0, sizeof(state.bass_velocity));
}

void retro_init(void)
{}

void retro_deinit(void)
{}

unsigned retro_api_version(void)
{
   return RETRO_API_VERSION;
}

void retro_set_controller_port_device(unsigned port, unsigned device)
{
   (void)port;
   (void)device;
}

void retro_get_system_info(struct retro_system_info *info)
{
   memset(info, 0, sizeof(*info));
   info->library_name     = "MIDIViz";
   info->library_version  = "v1";
   info->need_fullpath    = false;
   info->valid_extensions = "mid";
}

void retro_get_system_av_info(struct retro_system_av_info *info)
{
   info->timing.fps = 60.0;
   info->timing.sample_rate = 44100.0;

   info->geometry.base_width = BASE_WIDTH;
   info->geometry.base_height = BASE_HEIGHT;
   info->geometry.max_width = BASE_WIDTH;
   info->geometry.max_height = BASE_HEIGHT;
   info->geometry.aspect_ratio = (float)BASE_WIDTH / (float)BASE_HEIGHT;
}

static retro_video_refresh_t video_cb;
static retro_audio_sample_t audio_cb;
static retro_audio_sample_batch_t audio_batch_cb;
static retro_environment_t environ_cb;
static retro_input_poll_t input_poll_cb;
static retro_input_state_t input_state_cb;

void retro_set_environment(retro_environment_t cb)
{
   environ_cb = cb;

   bool no_rom = true;
   cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &no_rom);
}

void retro_set_audio_sample(retro_audio_sample_t cb)
{
   audio_cb = cb;
}

void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb)
{
   audio_batch_cb = cb;
}

void retro_set_input_poll(retro_input_poll_t cb)
{
   input_poll_cb = cb;
}

void retro_set_input_state(retro_input_state_t cb)
{
   input_state_cb = cb;
}

void retro_set_video_refresh(retro_video_refresh_t cb)
{
   video_cb = cb;
}

static uint32_t find_memory_type_from_requirements(
      uint32_t device_requirements, uint32_t host_requirements)
{
   const VkPhysicalDeviceMemoryProperties *props = &vk.memory_properties;
   for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; i++)
   {
      if (device_requirements & (1u << i))
      {
         if ((props->memoryTypes[i].propertyFlags & host_requirements) == host_requirements)
         {
            return i;
         }
      }
   }

   return 0;
}

static void pipeline_barrier(VkCommandBuffer cmd,
      VkPipelineStageFlags src_stage,
      VkPipelineStageFlags dst_stage,
      VkAccessFlags src_access,
      VkAccessFlags dst_access)
{
   const VkMemoryBarrier barrier = {
      VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      nullptr,
      src_access,
      dst_access,
   };

   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0,
         1, &barrier, 0, nullptr, 0, nullptr);
}

struct BaseGenerate
{
   uint32_t base;
   uint32_t mask;
};

static void generate_particles(VkCommandBuffer cmd, VkPipeline pipeline, unsigned count, BaseGenerate &base, size_t size)
{
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   base.base = vk.particle_ptr;
   base.mask = NUM_PARTICLES - 1u;

   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
         vk.compute_pipeline_layout, 0, 1, &vk.desc_set, 0, nullptr);
   vkCmdPushConstants(cmd, vk.compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
         0, size, &base);
   vkCmdDispatch(cmd, count / 64, 1, 1);

   vk.particle_ptr = (vk.particle_ptr + count) & (NUM_PARTICLES - 1);
}

static float fract(float v)
{
   return v - floor(v);
}

static void move_particles(VkCommandBuffer cmd, float step_frames)
{
   pipeline_barrier(cmd,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_ACCESS_SHADER_WRITE_BIT,
         VK_ACCESS_SHADER_WRITE_BIT);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk.move_pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
         vk.compute_pipeline_layout, 0, 1, &vk.desc_set, 0, nullptr);

   struct Push
   {
      uint32_t base = 0;
      uint32_t mask = ~0u;
      float delta;
      float frame;
      float kick;
      float kick_beat;
      float snare;
      float snare_beat;
   };

   float beat = midi_file.get_beat();

   Push push;
   push.delta = step_frames / 60.0f;
   push.frame = 2.0 * fract(0.25 * beat);
   push.kick = state.kick;
   push.kick_beat = 1.0f - (beat - state.kick_beat);
   push.snare = state.snare;
   push.snare_beat = beat - state.snare_beat;
   if (push.frame > 1.0)
      push.frame = 2.0 - push.frame;

   vkCmdPushConstants(cmd, vk.compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
         0, sizeof(push), &push);
   vkCmdDispatch(cmd, NUM_PARTICLES / 64, 1, 1);
}

enum Tracks
{
   BAND_PASS_TRANCE = 2,
   BASS = 3,
   DRUMS = 4,
   LEAD = 5,
   GATED = 6,
   PLUCKS = 7,
   PIANO = 8,
};

static void vulkan_update_particles(VkCommandBuffer cmd)
{
   double current_audio_frame = double(audio_frames.load(std::memory_order_relaxed)) / FRAMES;
   double diff_frames = current_audio_frame - state.frame - 1.0;
   double step_frames = 1.0 + 0.01 * diff_frames;

   unsigned last_frame = unsigned(state.frame);
   state.frame += step_frames;
   unsigned current_frame = unsigned(state.frame);

   for (unsigned i = last_frame; i < current_frame; i++)
   {
      midi_file.step();

      for (auto &e : midi_file.get_events(GATED))
      {
         if (e.kon && e.velocity)
         {
            state.r += 0.1f;
            state.g += 0.0f;
            state.b += 0.1f;
         }
      }

      for (auto &e : midi_file.get_events(DRUMS))
      {
         if (e.kon && e.velocity)
         {
            if (e.note == 36) // C1: Kick
            {
               state.kick_beat = midi_file.get_beat();
               state.kick += 1.0f;
            }
            else if (e.note == 37) // C#1: Snare
            {
               state.snare_beat = midi_file.get_beat();
               state.snare += 1.0f;
            }
            else
               state.g += 0.10f;
         }
      }

      state.kick -= state.kick * 0.05f;
      state.snare -= state.snare * 0.025f;

      for (auto &e : midi_file.get_events(LEAD))
      {
         if (e.kon && e.velocity)
         {
            state.lead[e.note] = true;
            state.last_lead[e.note] = state.frame;
            state.lead_velocity[e.note] = e.velocity * (1.0f / 127.0f);
         }
         else if (!e.kon || !e.velocity)
            state.lead[e.note] = false;
      }

      for (auto &e : midi_file.get_events(BASS))
      {
         if (e.kon && e.velocity)
         {
            state.bass[e.note] = true;
            state.last_bass[e.note] = state.frame;
            state.bass_velocity[e.note] = e.velocity * (1.0f / 127.0f);
         }
         else if (!e.kon || !e.velocity)
            state.bass[e.note] = false;
      }

      state.r -= state.r * 0.06f;
      state.g -= state.g * 0.07f;
      state.b -= state.b * 0.08f;

      // All compute stuff here.
      if (i == last_frame)
      {
         pipeline_barrier(cmd,
               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               0,
               0);
      }

      // Generate pluck particles.
      for (auto &e : midi_file.get_events(PLUCKS))
      {
         struct Pluck : BaseGenerate
         {
            float key;
            float vel;
         };

         if (e.kon && e.velocity)
         {
            Pluck gen;
            gen.key = float(e.note);
            gen.vel = float(e.velocity) * (1.0f / 127.0f);
            generate_particles(cmd, vk.pluck_pipeline, 256, gen, sizeof(gen));
         }
      }

      // Generate arp particles.
      for (auto &e : midi_file.get_events(BAND_PASS_TRANCE))
      {
         struct BandPass : BaseGenerate
         {
            float key;
            float vel;
         };

         if (e.kon && e.velocity)
         {
            BandPass gen;
            gen.key = float(e.note);
            gen.vel = float(e.velocity) * (1.0f / 127.0f);
            generate_particles(cmd, vk.arp_pipeline, 512, gen, sizeof(gen));
         }
      }

      for (auto &e : midi_file.get_events(PIANO))
      {
         struct Piano : BaseGenerate
         {
            float key;
            float vel;
         };

         if (e.kon && e.velocity)
         {
            Piano gen;
            gen.key = float(e.note);
            gen.vel = float(e.velocity) * (1.0f / 127.0f);
            generate_particles(cmd, vk.piano_pipeline, 256, gen, sizeof(gen));
         }
      }

      float beat_phase = 2.0f * fract(0.25f * midi_file.get_beat());
      if (beat_phase > 1.0f)
         beat_phase = 2.0f - beat_phase;
      beat_phase = (beat_phase - 0.5f) * 2.0f;

      for (unsigned i = 0; i < 128; i++)
      {
         struct Lead : BaseGenerate
         {
            float key;
            float vel;
            float phase;
         };

         if (state.lead[i])
         {
            Lead gen;
            gen.key = float(i);

            unsigned t = state.frame - state.last_lead[i];
            gen.vel = float(state.lead_velocity[i]) * exp2(float(t) * -0.005f);
            gen.phase = beat_phase;
            generate_particles(cmd, vk.lead_pipeline, 128, gen, sizeof(gen));
         }

         if (state.bass[i])
         {
            Lead gen;
            gen.key = float(i);

            unsigned t = state.frame - state.last_bass[i];
            gen.vel = float(state.bass_velocity[i]) * exp2(float(t) * -0.08f);
            gen.phase = beat_phase;
            generate_particles(cmd, vk.bass_pipeline, 256, gen, sizeof(gen));
         }
      }
   }

   move_particles(cmd, step_frames);

   pipeline_barrier(cmd,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
         VK_ACCESS_SHADER_WRITE_BIT,
         VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
}

static void vulkan_render(void)
{
   VkCommandBuffer cmd = vk.cmd[vk.index];

   VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   vkResetCommandBuffer(cmd, 0);
   vkBeginCommandBuffer(cmd, &begin_info);

   vulkan_update_particles(cmd);

   VkImageMemoryBarrier prepare_rendering = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
   prepare_rendering.srcAccessMask = 0;
   prepare_rendering.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
   prepare_rendering.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   prepare_rendering.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   prepare_rendering.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   prepare_rendering.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   prepare_rendering.image = vk.images[vk.index].create_info.image;
   prepare_rendering.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   prepare_rendering.subresourceRange.levelCount = 1;
   prepare_rendering.subresourceRange.layerCount = 1;
   vkCmdPipelineBarrier(cmd,
         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         false, 
         0, nullptr,
         0, nullptr,
         1, &prepare_rendering);

   VkClearValue clear_value;
   clear_value.color.float32[0] = state.r;
   clear_value.color.float32[1] = state.g;
   clear_value.color.float32[2] = state.b;
   clear_value.color.float32[3] = 1.0f;

   VkRenderPassBeginInfo rp_begin = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
   rp_begin.renderPass = vk.render_pass;
   rp_begin.framebuffer = vk.framebuffers[vk.index];
   rp_begin.renderArea.extent.width = width;
   rp_begin.renderArea.extent.height = height;
   rp_begin.clearValueCount = 1;
   rp_begin.pClearValues = &clear_value;
   vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

   auto set_viewport = [](VkCommandBuffer cmd, unsigned width, unsigned height) {
      VkViewport vp = { 0 };
      vp.x = 0.0f;
      vp.y = 0.0f;
      vp.width = width;
      vp.height = height;
      vp.minDepth = 0.0f;
      vp.maxDepth = 1.0f;
      vkCmdSetViewport(cmd, 0, 1, &vp);

      VkRect2D scissor;
      memset(&scissor, 0, sizeof(scissor));
      scissor.extent.width = width;
      scissor.extent.height = height;
      vkCmdSetScissor(cmd, 0, 1, &scissor);
   };

   VkDeviceSize offset = 0;

   // Particles
   {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.particle_pipeline);
      set_viewport(cmd, width, height);
      vkCmdBindVertexBuffers(cmd, 0, 1, &vk.positions.buffer, &offset);
      vkCmdBindVertexBuffers(cmd, 1, 1, &vk.color.buffer, &offset);

      struct Push
      {
         float scale[2];
         float point_scale;
      };
      Push push = { { float(height) / width, 1.0f }, float(BASE_WIDTH) / 640.0f };
      vkCmdPushConstants(cmd, vk.pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(push), &push);

      vkCmdDraw(cmd, NUM_PARTICLES, 1, 0, 0);
   }

   // Kick
   if (state.kick > 0.001f)
   {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.kick_pipeline);
      set_viewport(cmd, width, height);
      vkCmdBindVertexBuffers(cmd, 0, 1, &vk.vbo.buffer, &offset);

      struct Push
      {
         float vel;
      };
      Push push;
      push.vel = state.kick;
      vkCmdPushConstants(cmd, vk.pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(push), &push);
      vkCmdDraw(cmd, 4, 1, 0, 0);
   }

   if (state.snare > 0.001f)
   {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.snare_pipeline);
      set_viewport(cmd, width, height);
      vkCmdBindVertexBuffers(cmd, 0, 1, &vk.vbo.buffer, &offset);

      struct Push
      {
         float vel;
      };
      Push push;
      push.vel = state.snare;
      vkCmdPushConstants(cmd, vk.pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(push), &push);
      vkCmdDraw(cmd, 4, 1, 0, 0);
   }

   vkCmdEndRenderPass(cmd);

   VkImageMemoryBarrier prepare_presentation = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
   prepare_presentation.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   prepare_presentation.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   prepare_presentation.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   prepare_presentation.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

   prepare_presentation.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   prepare_presentation.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   prepare_presentation.image = vk.images[vk.index].create_info.image;
   prepare_presentation.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   prepare_presentation.subresourceRange.levelCount = 1;
   prepare_presentation.subresourceRange.layerCount = 1;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
         false,
         0, nullptr,
         0, nullptr,
         1, &prepare_presentation);

   vkEndCommandBuffer(cmd);
}

static struct buffer create_buffer(const void *initial, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
   struct buffer buffer;
   VkDevice device = vulkan->device;

   VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
   info.usage = usage;
   info.size = size;

   vkCreateBuffer(device, &info, nullptr, &buffer.buffer);

   VkMemoryRequirements mem_reqs;
   vkGetBufferMemoryRequirements(device, buffer.buffer, &mem_reqs);

   VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
   alloc.allocationSize = mem_reqs.size;

   alloc.memoryTypeIndex = find_memory_type_from_requirements(mem_reqs.memoryTypeBits,
         properties);

   vkAllocateMemory(device, &alloc, nullptr, &buffer.memory);
   vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0);

   if (initial)
   {
      void *ptr;
      vkMapMemory(device, buffer.memory, 0, size, 0, &ptr);
      memcpy(ptr, initial, size);
      vkUnmapMemory(device, buffer.memory);
   }

   return buffer;
}

static void init_buffers(void)
{
   static const float data[] = {
      -1.0f, -1.0f,
      -1.0f, +1.0f,
      +1.0f, -1.0f,
      +1.0f, +1.0f,
   };
   vk.vbo = create_buffer(data, sizeof(data), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   vk.positions = create_buffer(nullptr, NUM_PARTICLES * 2 * sizeof(float),
         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   vk.velocity = create_buffer(nullptr, NUM_PARTICLES * 2 * sizeof(uint16_t),
         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   vk.color = create_buffer(nullptr, NUM_PARTICLES * 4 * sizeof(uint16_t),
         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

   auto cmd = vk.cmd[0];
   VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   vkResetCommandBuffer(cmd, 0);
   vkBeginCommandBuffer(cmd, &begin_info);
   vkCmdFillBuffer(cmd, vk.positions.buffer, 0, NUM_PARTICLES * 2 * sizeof(float), 0);
   vkCmdFillBuffer(cmd, vk.velocity.buffer, 0, NUM_PARTICLES * 2 * sizeof(uint16_t), 0);
   vkCmdFillBuffer(cmd, vk.color.buffer, 0, NUM_PARTICLES * 4 * sizeof(uint16_t), 0);
   vkEndCommandBuffer(cmd);

   VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
   submit.commandBufferCount = 1;
   submit.pCommandBuffers = &cmd;
   vulkan->lock_queue(vulkan->handle);
   vkQueueSubmit(vulkan->queue, 1, &submit, VK_NULL_HANDLE);
   vulkan->unlock_queue(vulkan->handle);

   vkQueueWaitIdle(vulkan->queue);
}

static VkShaderModule create_shader_module(const uint32_t *data, size_t size)
{
   VkShaderModuleCreateInfo module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
   VkShaderModule module;
   module_info.codeSize = size;
   module_info.pCode = data;
   vkCreateShaderModule(vulkan->device, &module_info, nullptr, &module);
   return module;
}

static void init_descriptor(void)
{
   VkDevice device = vulkan->device;

   VkDescriptorSetLayoutBinding bindings[3] = {};
   for (unsigned i = 0; i < 3; i++)
   {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
   }

   const VkDescriptorPoolSize pool_sizes[3] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };

   VkDescriptorSetLayoutCreateInfo set_layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
   set_layout_info.bindingCount = 3;
   set_layout_info.pBindings = bindings;
   vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &vk.set_layout);

   VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
   pool_info.maxSets = 1;
   pool_info.poolSizeCount = 3;
   pool_info.pPoolSizes = pool_sizes;
   vkCreateDescriptorPool(device, &pool_info, nullptr, &vk.desc_pool);

   VkPipelineLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };

   VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
   alloc_info.descriptorPool = vk.desc_pool;
   alloc_info.descriptorSetCount = 1;
   alloc_info.pSetLayouts = &vk.set_layout;

   vkAllocateDescriptorSets(device, &alloc_info, &vk.desc_set);

   VkWriteDescriptorSet writes[3] = {
      { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
      { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
      { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
   };
   VkDescriptorBufferInfo buffer_infos[3] = {};

   for (unsigned i = 0; i < 3; i++)
   {
      writes[i].dstSet = vk.desc_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_infos[i];
   }

   buffer_infos[0].buffer = vk.positions.buffer;
   buffer_infos[0].range = NUM_PARTICLES * 2 * sizeof(float);
   buffer_infos[1].buffer = vk.velocity.buffer;
   buffer_infos[1].range = NUM_PARTICLES * 2 * sizeof(uint16_t);
   buffer_infos[2].buffer = vk.color.buffer;
   buffer_infos[2].range = NUM_PARTICLES * 4 * sizeof(uint16_t);

   vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

   // Particle pipeline
   static const VkPushConstantRange ranges[1] = {
      {
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
         0, 64,
      },
   };
   layout_info.pushConstantRangeCount = 1;
   layout_info.pPushConstantRanges = ranges;

   layout_info.setLayoutCount = 0;
   vkCreatePipelineLayout(device, &layout_info, nullptr, &vk.pipeline_layout);
   ///

   // Compute pipeline
   static const VkPushConstantRange compute_range = {
      VK_SHADER_STAGE_COMPUTE_BIT,
      0, 128,
   };
   layout_info.pushConstantRangeCount = 1;
   layout_info.pPushConstantRanges = &compute_range;

   layout_info.setLayoutCount = 1;
   layout_info.pSetLayouts = &vk.set_layout;
   vkCreatePipelineLayout(device, &layout_info, nullptr, &vk.compute_pipeline_layout);
   ///
}

static void init_generation_pipeline()
{
   VkDevice device = vulkan->device;
   VkComputePipelineCreateInfo pipe = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
   pipe.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   pipe.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
   pipe.stage.pName = "main";
   pipe.layout = vk.compute_pipeline_layout;

   static const uint32_t generate_comp[] =
#include "shaders/generate.comp.inc"
      ;

   static const uint32_t pluck_comp[] =
#include "shaders/pluck.comp.inc"
      ;

   static const uint32_t move_comp[] =
#include "shaders/move.comp.inc"
      ;

   static const uint32_t arp_comp[] =
#include "shaders/arp.comp.inc"
      ;

   static const uint32_t bass_comp[] =
#include "shaders/bass.comp.inc"
      ;

   static const uint32_t lead_comp[] =
#include "shaders/lead.comp.inc"
      ;

   static const uint32_t piano_comp[] =
#include "shaders/piano.comp.inc"
      ;

#define BUILD(x) \
   pipe.stage.module = create_shader_module(x##_comp, sizeof(x##_comp)); \
   vkCreateComputePipelines(device, vk.pipeline_cache, \
         1, &pipe, nullptr, &vk.x##_pipeline); \
   vkDestroyShaderModule(device, pipe.stage.module, nullptr)

   BUILD(generate);
   BUILD(move);
   BUILD(pluck);
   BUILD(arp);
   BUILD(bass);
   BUILD(lead);
   BUILD(piano);
}

static void init_quad_pipeline(VkPipeline &pipeline,
      const uint32_t *vert, size_t vert_size,
      const uint32_t *frag, size_t frag_size)
{
   VkDevice device = vulkan->device;

   VkPipelineInputAssemblyStateCreateInfo input_assembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
   input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

   VkVertexInputAttributeDescription attributes[1] = {};
   attributes[0].location = 0;
   attributes[0].binding = 0;
   attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
   attributes[0].offset = 0;

   VkVertexInputBindingDescription bindings[1] = {};
   bindings[0].binding = 0;
   bindings[0].stride = 8;
   bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

   VkPipelineVertexInputStateCreateInfo vertex_input = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   vertex_input.vertexBindingDescriptionCount = 1;
   vertex_input.pVertexBindingDescriptions = bindings;
   vertex_input.vertexAttributeDescriptionCount = 1;
   vertex_input.pVertexAttributeDescriptions = attributes;

   VkPipelineRasterizationStateCreateInfo raster = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
   raster.polygonMode = VK_POLYGON_MODE_FILL;
   raster.cullMode = VK_CULL_MODE_NONE;
   raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   raster.depthClampEnable = false;
   raster.rasterizerDiscardEnable = false;
   raster.depthBiasEnable = false;
   raster.lineWidth = 1.0f;

   VkPipelineColorBlendAttachmentState blend_attachment = {};
   blend_attachment.blendEnable = true;
   blend_attachment.colorWriteMask = 0xf;
   blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
   blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
   blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
   blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
   blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
   blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

   VkPipelineColorBlendStateCreateInfo blend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
   blend.attachmentCount = 1;
   blend.pAttachments = &blend_attachment;

   VkPipelineViewportStateCreateInfo viewport = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
   viewport.viewportCount = 1;
   viewport.scissorCount = 1;

   VkPipelineDepthStencilStateCreateInfo depth_stencil = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
   depth_stencil.depthTestEnable = false;
   depth_stencil.depthWriteEnable = false;
   depth_stencil.depthBoundsTestEnable = false;
   depth_stencil.stencilTestEnable = false;

   VkPipelineMultisampleStateCreateInfo multisample = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
   multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

   static const VkDynamicState dynamics[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
   };
   VkPipelineDynamicStateCreateInfo dynamic = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
   dynamic.pDynamicStates = dynamics;
   dynamic.dynamicStateCount = sizeof(dynamics) / sizeof(dynamics[0]);

   VkPipelineShaderStageCreateInfo shader_stages[2] = {
      { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
      { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
   };

   shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
   shader_stages[0].module = create_shader_module(vert, vert_size);
   shader_stages[0].pName = "main";
   shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
   shader_stages[1].module = create_shader_module(frag, frag_size); 
   shader_stages[1].pName = "main";

   VkGraphicsPipelineCreateInfo pipe = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
   pipe.stageCount = 2;
   pipe.pStages = shader_stages;
   pipe.pVertexInputState = &vertex_input;
   pipe.pInputAssemblyState = &input_assembly;
   pipe.pRasterizationState = &raster;
   pipe.pColorBlendState = &blend;
   pipe.pMultisampleState = &multisample;
   pipe.pViewportState = &viewport;
   pipe.pDepthStencilState = &depth_stencil;
   pipe.pDynamicState = &dynamic;
   pipe.renderPass = vk.render_pass;
   pipe.layout = vk.pipeline_layout;

   vkCreateGraphicsPipelines(device, vk.pipeline_cache, 1, &pipe, nullptr, &pipeline);
   vkDestroyShaderModule(device, shader_stages[0].module, nullptr);
   vkDestroyShaderModule(device, shader_stages[1].module, nullptr);
}

static void init_kick_pipelines()
{
   static const uint32_t kick_vert[] =
#include "shaders/kick.vert.inc"
      ;

   static const uint32_t kick_frag[] =
#include "shaders/kick.frag.inc"
      ;

   static const uint32_t snare_vert[] =
#include "shaders/snare.vert.inc"
      ;

   static const uint32_t snare_frag[] =
#include "shaders/snare.frag.inc"
      ;

   init_quad_pipeline(vk.kick_pipeline,
         kick_vert, sizeof(kick_vert),
         kick_frag, sizeof(kick_frag));

   init_quad_pipeline(vk.snare_pipeline,
         snare_vert, sizeof(snare_vert),
         snare_frag, sizeof(snare_frag));
}

static void init_particle_pipeline()
{
   VkDevice device = vulkan->device;

   VkPipelineInputAssemblyStateCreateInfo input_assembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
   input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

   VkVertexInputAttributeDescription attributes[2] = {};
   attributes[0].location = 0;
   attributes[0].binding = 0;
   attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
   attributes[0].offset = 0;
   attributes[1].location = 1;
   attributes[1].binding = 1;
   attributes[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
   attributes[1].offset = 0;

   VkVertexInputBindingDescription bindings[2] = {};
   bindings[0].binding = 0;
   bindings[0].stride = 8;
   bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
   bindings[1].binding = 1;
   bindings[1].stride = 8;
   bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

   VkPipelineVertexInputStateCreateInfo vertex_input = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   vertex_input.vertexBindingDescriptionCount = 2;
   vertex_input.pVertexBindingDescriptions = bindings;
   vertex_input.vertexAttributeDescriptionCount = 2;
   vertex_input.pVertexAttributeDescriptions = attributes;

   VkPipelineRasterizationStateCreateInfo raster = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
   raster.polygonMode = VK_POLYGON_MODE_FILL;
   raster.cullMode = VK_CULL_MODE_NONE;
   raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   raster.depthClampEnable = false;
   raster.rasterizerDiscardEnable = false;
   raster.depthBiasEnable = false;
   raster.lineWidth = 1.0f;

   VkPipelineColorBlendAttachmentState blend_attachment = {};
   blend_attachment.blendEnable = true;
   blend_attachment.colorWriteMask = 0xf;
   blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
   blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
   blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
   blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
   blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
   blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

   VkPipelineColorBlendStateCreateInfo blend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
   blend.attachmentCount = 1;
   blend.pAttachments = &blend_attachment;

   VkPipelineViewportStateCreateInfo viewport = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
   viewport.viewportCount = 1;
   viewport.scissorCount = 1;

   VkPipelineDepthStencilStateCreateInfo depth_stencil = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
   depth_stencil.depthTestEnable = false;
   depth_stencil.depthWriteEnable = false;
   depth_stencil.depthBoundsTestEnable = false;
   depth_stencil.stencilTestEnable = false;

   VkPipelineMultisampleStateCreateInfo multisample = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
   multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

   static const VkDynamicState dynamics[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
   };
   VkPipelineDynamicStateCreateInfo dynamic = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
   dynamic.pDynamicStates = dynamics;
   dynamic.dynamicStateCount = sizeof(dynamics) / sizeof(dynamics[0]);

   VkPipelineShaderStageCreateInfo shader_stages[2] = {
      { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
      { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
   };

   static const uint32_t particle_vert[] =
#include "shaders/particle.vert.inc"
      ;

   static const uint32_t particle_frag[] =
#include "shaders/particle.frag.inc"
      ;

   shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
   shader_stages[0].module = create_shader_module(particle_vert, sizeof(particle_vert));
   shader_stages[0].pName = "main";
   shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
   shader_stages[1].module = create_shader_module(particle_frag, sizeof(particle_frag));
   shader_stages[1].pName = "main";

   VkGraphicsPipelineCreateInfo pipe = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
   pipe.stageCount = 2;
   pipe.pStages = shader_stages;
   pipe.pVertexInputState = &vertex_input;
   pipe.pInputAssemblyState = &input_assembly;
   pipe.pRasterizationState = &raster;
   pipe.pColorBlendState = &blend;
   pipe.pMultisampleState = &multisample;
   pipe.pViewportState = &viewport;
   pipe.pDepthStencilState = &depth_stencil;
   pipe.pDynamicState = &dynamic;
   pipe.renderPass = vk.render_pass;
   pipe.layout = vk.pipeline_layout;

   vkCreateGraphicsPipelines(device, vk.pipeline_cache, 1, &pipe, nullptr, &vk.particle_pipeline);
   vkDestroyShaderModule(device, shader_stages[0].module, nullptr);
   vkDestroyShaderModule(device, shader_stages[1].module, nullptr);
}

static void init_pipelines(void)
{
   init_particle_pipeline();
   init_kick_pipelines();
   init_generation_pipeline();
}

static void init_render_pass(VkFormat format)
{
   VkAttachmentDescription attachment = { 0 };
   attachment.format = format;
   attachment.samples = VK_SAMPLE_COUNT_1_BIT;
   attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

   attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

   VkAttachmentReference color_ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription subpass = { 0 };
   subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   subpass.colorAttachmentCount = 1;
   subpass.pColorAttachments = &color_ref;

   VkRenderPassCreateInfo rp_info = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
   rp_info.attachmentCount = 1;
   rp_info.pAttachments = &attachment;
   rp_info.subpassCount = 1;
   rp_info.pSubpasses = &subpass;
   vkCreateRenderPass(vulkan->device, &rp_info, nullptr, &vk.render_pass);
}

static void init_swapchain(void)
{
   VkDevice device = vulkan->device;

   for (unsigned i = 0; i < vk.num_swapchain_images; i++)
   {
      VkImageCreateInfo image = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
      image.extent.width = width;
      image.extent.height = height;
      image.extent.depth = 1;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage =
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
         VK_IMAGE_USAGE_SAMPLED_BIT |
         VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image.mipLevels = 1;
      image.arrayLayers = 1;

      vkCreateImage(device, &image, nullptr, &vk.images[i].create_info.image);

      VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
      VkMemoryRequirements mem_reqs;

      vkGetImageMemoryRequirements(device, vk.images[i].create_info.image, &mem_reqs);
      alloc.allocationSize = mem_reqs.size;
      alloc.memoryTypeIndex = find_memory_type_from_requirements(
            mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      vkAllocateMemory(device, &alloc, nullptr, &vk.image_memory[i]);
      vkBindImageMemory(device, vk.images[i].create_info.image, vk.image_memory[i], 0);

      vk.images[i].create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      vk.images[i].create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vk.images[i].create_info.format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
      vk.images[i].create_info.subresourceRange.baseMipLevel = 0;
      vk.images[i].create_info.subresourceRange.baseArrayLayer = 0;
      vk.images[i].create_info.subresourceRange.levelCount = 1;
      vk.images[i].create_info.subresourceRange.layerCount = 1;
      vk.images[i].create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      vk.images[i].create_info.components.r = VK_COMPONENT_SWIZZLE_R;
      vk.images[i].create_info.components.g = VK_COMPONENT_SWIZZLE_G;
      vk.images[i].create_info.components.b = VK_COMPONENT_SWIZZLE_B;
      vk.images[i].create_info.components.a = VK_COMPONENT_SWIZZLE_A;

      vkCreateImageView(device, &vk.images[i].create_info,
            nullptr, &vk.images[i].image_view);
      vk.images[i].image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      VkFramebufferCreateInfo fb_info = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
      fb_info.renderPass = vk.render_pass;
      fb_info.attachmentCount = 1;
      fb_info.pAttachments = &vk.images[i].image_view;
      fb_info.width = width;
      fb_info.height = height;
      fb_info.layers = 1;

      vkCreateFramebuffer(device, &fb_info, nullptr, &vk.framebuffers[i]);
   }
}

static void init_command(void)
{
   VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
   VkCommandBufferAllocateInfo info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };

   pool_info.queueFamilyIndex = vulkan->queue_index;
   pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

   for (unsigned i = 0; i < vk.num_swapchain_images; i++)
   {
      vkCreateCommandPool(vulkan->device, &pool_info, nullptr, &vk.cmd_pool[i]);
      info.commandPool = vk.cmd_pool[i];
      info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      info.commandBufferCount = 1;
      vkAllocateCommandBuffers(vulkan->device, &info, &vk.cmd[i]);
   }
}

static void vulkan_test_init(void)
{
   vkGetPhysicalDeviceProperties(vulkan->gpu, &vk.gpu_properties);
   vkGetPhysicalDeviceMemoryProperties(vulkan->gpu, &vk.memory_properties);

   unsigned num_images = 0;
   uint32_t mask = vulkan->get_sync_index_mask(vulkan->handle);
   for (unsigned i = 0; i < 32; i++)
      if (mask & (1u << i))
         num_images = i + 1;
   vk.num_swapchain_images = num_images;
   vk.swapchain_mask = mask;

   init_command();
   init_buffers();
   init_descriptor();

   VkPipelineCacheCreateInfo pipeline_cache_info = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
   vkCreatePipelineCache(vulkan->device, &pipeline_cache_info,
         nullptr, &vk.pipeline_cache);

   init_render_pass(VK_FORMAT_A2R10G10B10_UNORM_PACK32);
   init_pipelines();
   init_swapchain();
}

static void vulkan_test_deinit(void)
{
   if (!vulkan)
      return;

   VkDevice device = vulkan->device;
   vkDeviceWaitIdle(device);

   for (unsigned i = 0; i < vk.num_swapchain_images; i++)
   {
      vkDestroyFramebuffer(device, vk.framebuffers[i], nullptr);
      vkDestroyImageView(device, vk.images[i].image_view, nullptr);
      vkFreeMemory(device, vk.image_memory[i], nullptr);
      vkDestroyImage(device, vk.images[i].create_info.image, nullptr);
   }

   vkDestroyDescriptorPool(device, vk.desc_pool, nullptr);
   vkDestroyDescriptorSetLayout(device, vk.set_layout, nullptr);
   vkDestroyRenderPass(device, vk.render_pass, nullptr);

   vkDestroyPipelineLayout(device, vk.pipeline_layout, nullptr);
   vkDestroyPipelineLayout(device, vk.compute_pipeline_layout, nullptr);

   vkDestroyPipeline(device, vk.particle_pipeline, nullptr);
   vkDestroyPipeline(device, vk.generate_pipeline, nullptr);
   vkDestroyPipeline(device, vk.move_pipeline, nullptr);
   vkDestroyPipeline(device, vk.pluck_pipeline, nullptr);
   vkDestroyPipeline(device, vk.kick_pipeline, nullptr);
   vkDestroyPipeline(device, vk.snare_pipeline, nullptr);
   vkDestroyPipeline(device, vk.arp_pipeline, nullptr);
   vkDestroyPipeline(device, vk.bass_pipeline, nullptr);
   vkDestroyPipeline(device, vk.lead_pipeline, nullptr);
   vkDestroyPipeline(device, vk.piano_pipeline, nullptr);

   free_buffer(device, &vk.vbo);
   free_buffer(device, &vk.positions);
   free_buffer(device, &vk.velocity);
   free_buffer(device, &vk.color);

   vkDestroyPipelineCache(device, vk.pipeline_cache, nullptr);

   for (unsigned i = 0; i < vk.num_swapchain_images; i++)
   {
      vkFreeCommandBuffers(device, vk.cmd_pool[i], 1, &vk.cmd[i]);
      vkDestroyCommandPool(device, vk.cmd_pool[i], nullptr);
   }

   memset(&vk, 0, sizeof(vk));
}

static void audio_set_state(bool enable)
{
   audio_cb_enable.store(enable);
}

static void audio_callback()
{
   int16_t buffer[FRAMES * 2];

   audio_lock.lock();
   auto frames = sf_readf_short(audio_file, buffer, FRAMES);
   audio_frames.fetch_add(FRAMES, std::memory_order_relaxed);
   audio_lock.unlock();

   if (frames)
      audio_batch_cb(buffer, frames);
}

void retro_run(void)
{
   // Looping
   if (midi_file.eof())
   {
      state.end_counter++;
      if (state.end_counter > 100)
      {
         struct retro_message seek_msg = {
            "Looping", 180,
         };
         environ_cb(RETRO_ENVIRONMENT_SET_MESSAGE, &seek_msg);
         retro_reset();
      }
   }

   // If we don't have cb interface, pump audio from here.
   if (!use_audio_cb)
      audio_callback();

   input_poll_cb();

   // Seeking
   bool right = input_state_cb(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_RIGHT);
   if (!state.right && right)
   {
      state.right = true;
      std::lock_guard<std::mutex> holder{audio_lock};

      auto frame = sf_seek(audio_file, FRAMES * 10 * 60, SEEK_CUR);

      if (frame >= 0)
      {
         audio_frames = frame;
         unsigned tick = frame / FRAMES;
         state.frame = tick;
         midi_file.seek(tick);

         char msg[256];
         sprintf(msg, "Seeking forward to %.2f s.", double(tick) / 60.0);
         struct retro_message seek_msg = {
            msg, 180,
         };
         environ_cb(RETRO_ENVIRONMENT_SET_MESSAGE, &seek_msg);
         reset_state();
      }
   }
   else if (!right)
      state.right = false;

   bool left = input_state_cb(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_LEFT);
   if (!state.left && left)
   {
      state.left = true;
      std::lock_guard<std::mutex> holder{audio_lock};

      auto frame = sf_seek(audio_file, FRAMES * -10 * 60, SEEK_CUR);
      if (frame == -1)
         frame = sf_seek(audio_file, 0, SEEK_SET);

      if (frame >= 0)
      {
         audio_frames = frame;
         unsigned tick = frame / FRAMES;
         midi_file.seek(tick);
         state.frame = tick;

         char msg[256];
         sprintf(msg, "Seeking backwards to %.2f s.", double(tick) / 60.0);
         struct retro_message seek_msg = {
            msg, 180,
         };
         environ_cb(RETRO_ENVIRONMENT_SET_MESSAGE, &seek_msg);
         reset_state();
      }
   }
   else if (!left)
      state.left = false;

   /* Very lazy way to do this. */
   if (vulkan->get_sync_index_mask(vulkan->handle) != vk.swapchain_mask)
   {
      vulkan_test_deinit();
      vulkan_test_init();
   }

   vulkan->wait_sync_index(vulkan->handle);

   vk.index = vulkan->get_sync_index(vulkan->handle);
   vulkan_render();
   vulkan->set_image(vulkan->handle, &vk.images[vk.index], 0, nullptr, VK_QUEUE_FAMILY_IGNORED);
   vulkan->set_command_buffers(vulkan->handle, 1, &vk.cmd[vk.index]);
   video_cb(RETRO_HW_FRAME_BUFFER_VALID, width, height, 0);
}

static void context_reset(void)
{
   fprintf(stderr, "Context reset!\n");
   if (!environ_cb(RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE, (void**)&vulkan) || !vulkan)
   {
      fprintf(stderr, "Failed to get HW rendering interface!\n");
      return;
   }

   if (vulkan->interface_version != RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION)
   {
      fprintf(stderr, "HW render interface mismatch, expected %u, got %u!\n",
            RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION, vulkan->interface_version);
      vulkan = nullptr;
      return;
   }

   vulkan_symbol_wrapper_init(vulkan->get_instance_proc_addr);
   vulkan_symbol_wrapper_load_core_instance_symbols(vulkan->instance);
   vulkan_symbol_wrapper_load_core_device_symbols(vulkan->device);
   vulkan_test_init();
}

static void context_destroy(void)
{
   fprintf(stderr, "Context destroy!\n");
   vulkan_test_deinit();
   vulkan = nullptr;
   memset(&vk, 0, sizeof(vk));
}

static const VkApplicationInfo *get_application_info(void)
{
   static const VkApplicationInfo info = {
      VK_STRUCTURE_TYPE_APPLICATION_INFO,
      nullptr,
      "MIDIViz",
      0,
      "libretro",
      0,
      VK_MAKE_VERSION(1, 0, 18),
   };
   return &info;
}

static bool retro_init_hw_context(void)
{
   hw_render.context_type = RETRO_HW_CONTEXT_VULKAN;
   hw_render.version_major = VK_MAKE_VERSION(1, 0, 18);
   hw_render.version_minor = 0;
   hw_render.context_reset = context_reset;
   hw_render.context_destroy = context_destroy;
   hw_render.cache_context = false;
   if (!environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw_render))
      return false;

   static const struct retro_hw_render_context_negotiation_interface_vulkan iface = {
      RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN,
      RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION,

      get_application_info,
      nullptr,
   };

   environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE, (void*)&iface);

   return true;
}

bool retro_load_game(const struct retro_game_info *info)
{
   if (!retro_init_hw_context())
   {
      fprintf(stderr, "HW Context could not be initialized, exiting...\n");
      return false;
   }

   midi_file = MIDI::File(60.0, "Forever Summer.mid");

   SF_INFO sf_info;
   audio_file = sf_open("Forever Summer.wav", SFM_READ, &sf_info);
   if (!audio_file)
      return false;

   if (sf_info.samplerate != 44100 || sf_info.channels != 2)
   {
      sf_close(audio_file);
      audio_file = nullptr;
      return false;
   }

   static struct retro_audio_callback cb = {
      audio_callback,
      audio_set_state,
   };
   use_audio_cb = environ_cb(RETRO_ENVIRONMENT_SET_AUDIO_CALLBACK, &cb);
   if (!use_audio_cb)
      audio_cb_enable.store(true);

   retro_reset();

   fprintf(stderr, "Loaded game!\n");
   (void)info;
   return true;
}

void retro_unload_game(void)
{
   if (audio_file)
      sf_close(audio_file);
   audio_file = nullptr;
}

unsigned retro_get_region(void)
{
   return RETRO_REGION_NTSC;
}

bool retro_load_game_special(unsigned type, const struct retro_game_info *info, size_t num)
{
   (void)type;
   (void)info;
   (void)num;
   return false;
}

size_t retro_serialize_size(void)
{
   return 0;
}

bool retro_serialize(void *data, size_t size)
{
   (void)data;
   (void)size;
   return false;
}

bool retro_unserialize(const void *data, size_t size)
{
   (void)data;
   (void)size;
   return false;
}

void *retro_get_memory_data(unsigned id)
{
   (void)id;
   return nullptr;
}

size_t retro_get_memory_size(unsigned id)
{
   (void)id;
   return 0;
}

void retro_reset(void)
{
   state.end_counter = 0;
   midi_file.reset();

   audio_lock.lock();
   if (audio_file)
      sf_seek(audio_file, 0, SEEK_SET);
   audio_frames = 0;
   audio_lock.unlock();

   state.frame = 0;
   reset_state();
}

void retro_cheat_reset(void)
{}

void retro_cheat_set(unsigned index, bool enabled, const char *code)
{
   (void)index;
   (void)enabled;
   (void)code;
}

