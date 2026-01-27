#include "javier/flash_controller.hpp"
#include <cstdio>

namespace javier {

/**
 * @brief GPU kernel implementation for Flash Bridge
 * 
 * This kernel processes flash bridge commands in parallel across
 * all bridge instances. Each thread handles one bridge.
 * Commands operate on specific channels within each bridge.
 */
__global__ void flash_bridge_kernel(
  FlashBridgeState* states,
  FlashBridgeCommand* commands,
  FlashBridgeResponse* responses,
  std::size_t num_bridges
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int>(num_bridges)) return;

  auto& bridge_state = states[idx];
  auto& cmd = commands[idx];
  auto& resp = responses[idx];

  // Reset response
  resp.success = false;
  resp.bytes_transferred = 0;
  resp.error_code = 0;
  resp.channel = cmd.channel;

  // Validate channel index
  if (cmd.channel >= bridge_state.num_channels) {
    resp.success = false;
    resp.error_code = 2; // Invalid channel
    return;
  }

  // Get the channel state for the target channel
  auto& channel = bridge_state.channels[cmd.channel];

  // Process command based on type
  switch (cmd.type) {
    case FlashBridgeCommand::Type::NOP:
      resp.success = true;
      break;

    case FlashBridgeCommand::Type::READ_PAGE:
      channel.operation = ChannelState::Operation::READ;
      channel.current_page = cmd.page;
      channel.current_block = cmd.block;
      channel.progress = 0;
      
      // Simulate reading data from NAND (in real implementation, would access flash hardware)
      // For demo purposes, fill with pattern data
      for (int i = 0; i < 256; ++i) {
        channel.data_buffer[i] = cmd.page * 256 + i + (cmd.channel * 10000);
      }
      
      channel.progress = 100;
      channel.operation = ChannelState::Operation::IDLE;
      resp.success = true;
      resp.bytes_transferred = 1024; // 256 * 4 bytes
      break;

    case FlashBridgeCommand::Type::WRITE_PAGE:
      channel.operation = ChannelState::Operation::WRITE;
      channel.current_page = cmd.page;
      channel.current_block = cmd.block;
      channel.progress = 0;
      
      // Simulate writing data to NAND (in real implementation, would write to flash hardware)
      channel.progress = 100;
      channel.operation = ChannelState::Operation::IDLE;
      resp.success = true;
      resp.bytes_transferred = 1024;
      break;

    case FlashBridgeCommand::Type::ERASE_BLOCK:
      channel.operation = ChannelState::Operation::ERASE;
      channel.current_block = cmd.block;
      channel.current_page = 0;
      channel.progress = 0;
      
      // Simulate erasing block in NAND (in real implementation, would erase flash hardware)
      // Clear data buffer - erased flash is typically 0xFF
      for (int i = 0; i < 256; ++i) {
        channel.data_buffer[i] = 0xFFFFFFFF;
      }
      
      channel.progress = 100;
      channel.operation = ChannelState::Operation::IDLE;
      resp.success = true;
      resp.bytes_transferred = 0;
      break;

    case FlashBridgeCommand::Type::PROGRAM_PAGE:
      channel.operation = ChannelState::Operation::PROGRAM;
      channel.current_page = cmd.page;
      channel.current_block = cmd.block;
      channel.progress = 0;
      
      // Simulate programming page in NAND
      channel.progress = 100;
      channel.operation = ChannelState::Operation::IDLE;
      resp.success = true;
      resp.bytes_transferred = 1024;
      break;

    case FlashBridgeCommand::Type::GET_STATUS:
      resp.success = true;
      resp.status = bridge_state.bridge_status;
      resp.bytes_transferred = 0;
      break;

    default:
      resp.success = false;
      resp.error_code = 1; // Unknown command
      break;
  }

  // Update channel status register
  channel.status = resp.success ? 0 : (1 << 0); // Bit 0 = error flag
  
  // Update bridge status if any channel has an error
  if (!resp.success) {
    bridge_state.bridge_status |= (1 << cmd.channel); // Set bit for this channel
  } else {
    bridge_state.bridge_status &= ~(1 << cmd.channel); // Clear bit for this channel
  }
}

/**
 * @brief Implementation of execute_kernel_impl for FlashBridge
 */
void FlashBridge::execute_kernel_impl(int blocks, int threads_per_block) {
  flash_bridge_kernel<<<blocks, threads_per_block>>>(
    get_states(), get_commands(), get_responses(), get_num_devices());
}

} // namespace javier
