# Javier - GPU-Based Hardware Control Framework

A template-based C++ framework for using GPU-like devices to control hardware. This framework provides a flexible, type-safe abstraction for building GPU-based firmware that can manage multiple hardware devices in parallel.

## Overview

Javier is designed to:
- Provide a template-based device abstraction layer
- Enable parallel hardware control via GPU kernels
- Support multiple device types through C++ templates
- Include a visual demo system for showcasing hardware operations

## Architecture

### Core Components

1. **`DeviceController<StateType, CommandType, ResponseType>`** - Template base class
   - Manages device state, commands, and responses
   - Handles GPU memory allocation (unified memory)
   - Provides execution framework for GPU kernels

2. **Device Types** - Specialized implementations
   - `FlashController` - Flash memory controller implementation
   - Easy to extend with new device types

3. **Visualizer** - Demo visualization system
   - Converts device states to visual representations
   - Provides ASCII-based visualization for demos
   - Can be extended with graphics libraries (OpenGL, Vulkan, etc.)

### Template Design

The framework uses C++ templates to provide:
- **Type Safety**: Compile-time checking of device state/command/response types
- **Performance**: Zero-overhead abstractions
- **Extensibility**: Easy to add new device types without modifying core framework

## Usage Example

```cpp
#include "javier/flash_controller.hpp"
#include "javier/visualizer.hpp"

// Create a flash controller managing 8 devices
javier::FlashController controller;
controller.initialize(8, 64, 4096); // 8 devices, 64 pages/block, 4096 bytes/page

// Queue operations
controller.queue_erase(0, 0);  // Erase block 0 on device 0
controller.queue_read(1, 10);  // Read page 10 on device 1
controller.queue_write(2, 5);  // Write page 5 on device 2

// Execute all commands in parallel on GPU
controller.execute();

// Visualize results
auto* states = controller.get_states();
for (size_t i = 0; i < 8; ++i) {
    auto visual = javier::DeviceVisualizer::visualize_flash_controller(states[i], i);
    javier::DeviceVisualizer::print_visual_state(visual);
}

// Cleanup
controller.cleanup();
```

## Building

```bash
# Configure (if needed)
cmake --preset wsl-debug

# Build
cmake --build build/wsl-debug

# Run demo
./build/wsl-debug/javier
```

## Extending the Framework

To add a new device type:

1. **Define State, Command, and Response structures:**
```cpp
struct MyDeviceState : public DeviceState {
    // Your device state fields
};

struct MyDeviceCommand : public DeviceCommand {
    enum class Type { /* ... */ } type;
    // Command parameters
};

struct MyDeviceResponse : public DeviceResponse {
    bool success;
    // Response data
};
```

2. **Create the device controller class:**
```cpp
class MyDeviceController : public DeviceController<
    MyDeviceState,
    MyDeviceCommand,
    MyDeviceResponse
> {
protected:
    void execute_kernel_impl(int blocks, int threads_per_block) override {
        my_device_kernel<<<blocks, threads_per_block>>>(
            get_states(), get_commands(), get_responses(), get_num_devices());
    }
};
```

3. **Implement the CUDA kernel:**
```cpp
__global__ void my_device_kernel(
    MyDeviceState* states,
    MyDeviceCommand* commands,
    MyDeviceResponse* responses,
    std::size_t num_devices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_devices) return;
    
    // Process command for device idx
    // Update states[idx], commands[idx], responses[idx]
}
```

## Flash Controller Demo

The included demo (`src/main.cu`) demonstrates:
- Parallel erase operations across multiple devices
- Parallel read operations
- Parallel write operations
- Status checking
- Visual state representation

## Requirements

- CUDA Toolkit (tested with CUDA 11+)
- CMake 3.24+
- C++17 compatible compiler
- NVIDIA GPU with compute capability 8.6+ (configurable in CMakeLists.txt)

## Notes

- The framework uses CUDA unified memory for seamless host/device access
- All device operations are executed in parallel on the GPU
- The visualizer can be extended to use graphics libraries for richer visualizations
- Template specialization ensures type safety while maintaining performance

## Future Enhancements

- [ ] Add more device types (SPI, I2C, GPIO controllers, etc.)
- [ ] Integrate with graphics libraries for real-time visualization
- [ ] Add device simulation mode for testing without hardware
- [ ] Performance profiling and optimization tools
- [ ] Multi-GPU support
