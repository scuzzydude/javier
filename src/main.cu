#include "javier/flash_controller.hpp"
#include "javier/visualizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <chrono>

/**
 * @brief Demo application showing GPU-based Flash Bridge framework
 * 
 * This demonstrates:
 * 1. Template-based device abstraction
 * 2. Parallel hardware control via GPU
 * 3. Bridge architecture with multiple NAND channels
 * 4. Visual state representation
 */
int main() {
  std::printf("=== Javier GPU Hardware Control Framework Demo ===\n");
  std::printf("Flash Bridge Demonstration (Bridges → Channels → NAND)\n\n");

  // Create flash bridges - each bridge connects to multiple NAND channels
  constexpr std::size_t num_bridges = 4;
  constexpr std::uint32_t channels_per_bridge = 8;
  javier::FlashBridge bridge;

  std::printf("Initializing %zu bridges, each with %u NAND channels...\n", 
              num_bridges, channels_per_bridge);
  if (!bridge.initialize(num_bridges, channels_per_bridge, 64, 4096)) {
    std::fprintf(stderr, "Failed to initialize flash bridges\n");
    return 1;
  }
  std::printf("Initialization complete!\n\n");

  // Demo sequence: perform various operations across bridges and channels
  std::printf("=== Demo Sequence ===\n\n");

  // Step 1: Erase blocks on various channels
  std::printf("Step 1: Erasing blocks on multiple channels...\n");
  for (std::size_t bridge_idx = 0; bridge_idx < num_bridges; ++bridge_idx) {
    for (std::uint32_t channel = 0; channel < channels_per_bridge; channel += 2) {
      bridge.queue_erase(bridge_idx, channel, bridge_idx * 10 + channel); // Erase block
    }
  }
  bridge.execute();
  
  // Visualize bridge states
  auto* states = bridge.get_states();
  for (std::size_t i = 0; i < num_bridges; ++i) {
    javier::DeviceVisualizer::visualize_bridge(states[i], i);
  }
  std::printf("\n");

  // Step 2: Read pages on various channels
  std::printf("Step 2: Reading pages from multiple channels...\n");
  for (std::size_t bridge_idx = 0; bridge_idx < num_bridges; ++bridge_idx) {
    for (std::uint32_t channel = 0; channel < channels_per_bridge; ++channel) {
      bridge.queue_read(bridge_idx, channel, bridge_idx * 100 + channel * 10);
    }
  }
  bridge.execute();
  
  // Visualize bridge states
  for (std::size_t i = 0; i < num_bridges; ++i) {
    javier::DeviceVisualizer::visualize_bridge(states[i], i);
  }
  std::printf("\n");

  // Step 3: Write pages on some channels
  std::printf("Step 3: Writing pages to selected channels...\n");
  for (std::size_t bridge_idx = 0; bridge_idx < num_bridges; ++bridge_idx) {
    for (std::uint32_t channel = 1; channel < channels_per_bridge; channel += 2) {
      bridge.queue_write(bridge_idx, channel, bridge_idx * 50 + channel * 5);
    }
  }
  bridge.execute();
  
  // Visualize bridge states
  for (std::size_t i = 0; i < num_bridges; ++i) {
    javier::DeviceVisualizer::visualize_bridge(states[i], i);
  }
  std::printf("\n");

  // Step 4: Check status of all bridges
  std::printf("Step 4: Checking status of all bridges...\n");
  for (std::size_t i = 0; i < num_bridges; ++i) {
    bridge.queue_status(i);
  }
  bridge.execute();
  
  auto* responses = bridge.get_responses();
  std::printf("\nStatus Summary:\n");
  for (std::size_t i = 0; i < num_bridges; ++i) {
    std::printf("  Bridge %zu: %s (error_code: %u)\n",
                i,
                responses[i].success ? "OK" : "ERROR",
                responses[i].error_code);
  }
  std::printf("\n");

  // Cleanup
  std::printf("Cleaning up...\n");
  bridge.cleanup();
  std::printf("Demo complete!\n");

  return 0;
}
