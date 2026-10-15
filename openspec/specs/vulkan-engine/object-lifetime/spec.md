# vulkan-engine/object-lifetime Specification

## Purpose
Ensures that every Vulkan object the engine creates is destroyed before the device it was created
from, so teardown is spec-valid and does not depend on driver tolerance.

## Requirements

### Requirement: Compute pipeline objects are destroyed with their owner
Every Vulkan object created to build or run a compute kernel SHALL be destroyed when the owning
kernel object is released. That covers the shader module, the pipeline, the pipeline layout, the
descriptor-set layout, any descriptor pool, and any pipeline cache.

#### Scenario: Dispatcher teardown
- **WHEN** a Vulkan dispatcher is destroyed
- **THEN** every compute-kernel object it created has been destroyed before the logical device is destroyed

### Requirement: Clean device teardown under validation
Destroying the engine SHALL NOT leave any device child object alive at `vkDestroyDevice`.

#### Scenario: Validation at exit
- **WHEN** any Vulkan test or benchmark binary exits normally with `BT_VK_VALIDATION=1`
- **THEN** the validation layer reports no `VUID-vkDestroyDevice-device-05137` (undestroyed child object) error

#### Scenario: Repeated engine construction
- **WHEN** an engine is created, destroyed and created again in the same process
- **THEN** both lifetimes complete without leaks or validation errors
