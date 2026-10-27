# tree-app/morton-codes Specification

## Purpose
Defines how tree points are quantized and encoded into 30-bit Morton codes, so every backend
produces identical codes for the same input, including at the edges of the coordinate range.

## Requirements

### Requirement: Quantization never wraps
Each coordinate SHALL be quantized to an integer in `[0, 1023]` before bit spreading. A coordinate at or
above the top of the range SHALL map to 1023, and one at or below the bottom SHALL map to 0.

#### Scenario: Point at the upper bound
- **WHEN** a point with a coordinate equal to `kMinCoord + kRange` is encoded
- **THEN** that axis quantizes to 1023, not 0

#### Scenario: Point just below the upper bound
- **WHEN** a coordinate's scaled value rounds to 1024.0 in float arithmetic
- **THEN** that axis quantizes to 1023

### Requirement: Every backend encodes identically
OMP, CUDA and Vulkan SHALL produce bit-identical Morton codes for the same points, including boundary
points.

#### Scenario: Boundary input on all backends
- **WHEN** the stage-1 differential test runs on an input that includes points at both range bounds
- **THEN** the OMP, CUDA and Vulkan codes are identical as multisets
