# SpyreCode Interface Specification

**Authors:**
* @ksarada
* @viji560
* @vswagath1989

## **Summary**
This document describes `SpyreCode` the artifacts produced by the Deeptools backend compiler for consumption by the torch-spyre device runtime to launch and execute a kernel (job) on Spyre.

## **Motivation**
`SpyreCode` is the contract between compiler and runtime to facilitate consumption of compilation artifacts for job launch during execution.

## **Proposed Implementation**

### Terminology and Backgroud Notes:
The execution of a computation kernel on the Spyre device is referred to as a <u>job</u>. The <u>computation kernel</u>  comprises of sequence of operations, uses dynamic shapes with input/output tensors resident on host or device. Job execution on Spyre involves a combination of executing programs on Spyre cores (using a compute control block) and transfers between host &#8660; Spyre (using a DMA control block). Jobs executed on Spyre use a (maximum) 128GB virtual address space, split into 8 segments each having a maximum length of 16GB.

### Components of `SpyreCode`
`SpyreCode` facilitates the runtime to execute a job on Spyre. The compiler produces a self-contained `JobPlan` — an ordered sequence of `RuntimeOperation` steps. Each step carries all the metadata it needs for execution: binary paths, correction metadata, device addresses, and allocation sizes. The three categories of information previously represented as standalone components (Job Binary, Host Compute Metadata) are now properties on individual steps within the JobPlan:

* **Job Binary**: Referenced by `binary_path` on the `ComputeOnDevice` step. The runtime (SpyreStream) is responsible for loading the binary to device during JobPlan loading.
* **Host Compute Metadata**: Embedded as `program_correction_metadata` on the `ProgramCorrection` step.
* **Allocation sizes**: Specified on the `ComputeOnDevice` step, telling the runtime how much space to allocate in segment 7 for the binary, correction data, and intermediate tensors.

Segment 7 is **reserved for use by `SpyreCode`**. This excludes segment 7 from being used for data tensors allocated by user-code (using .to() operation) or by compiler frontend (torch-inductor) generated code. The runtime manages segment 7 allocation internally — the JobPlan specifies the required sizes, and SpyreStream allocates the space via SpyreAllocator during loading.

### Job Plan

The Job plan is an ordered sequence of `RuntimeOperation` steps. The compiler produces the JobPlan directly as shared objects that the runtime consumes. Each step is self-contained, carrying all the metadata it needs for execution. The runtime executes the steps in sequence to complete the execution of the job.

The step types and their attributes are explained below:
* `ProgramCorrection`: A host-side operation that performs program correction. When a kernel uses symbolic addresses or shapes, the backend compiler produces correction metadata describing how symbol values map to corrections in the job binary. At runtime, this step takes resolved symbol values and the correction metadata as input and produces a correction tensor. Its attributes are:
  * `function`: The host function that performs program correction.
  * `program_correction_metadata`: Metadata needed by the host function for its processing (e.g., how input arguments/symbols must be interpreted in the context of the job binary — mapping a symbolic dimension size to a loop count correction in the binary). This is produced by the backend compiler as part of `SpyreCode`.

  Note: Unlike the previous `ComputeOnHost` command which carried explicit input/output handles and shapes (`ihandle`, `ishape`, `ohandle`, `oshape`), `ProgramCorrection` does not need these. The inputs to the host function (resolved symbol values — tensor addresses and shape values) are provided by the runtime (SpyreStream) at launch time from the `CompositeAddress` values on the SpyreTensors, not from the step itself. The output correction tensor buffer is pre-allocated by the runtime using the `breakdown_correctiondata` size from the associated `ComputeOnDevice` step. The `program_correction_metadata` tells the function how to interpret the inputs and produce the output of the correct size and layout.
* `ComputeOnDevice`: Triggers execution of computation on Spyre Cores. This is achieved by the runtime sending a control message to the card firmware which generates a compute control block (CB). Its attributes are:
  * `binary_path`: Path to the compiled binary produced by the backend compiler. This single binary may internally contain both a program correction program and the compute program.
  * `expected_input_shapes`: The compiled tensor shapes, used by the runtime to detect tiling requirements (when actual tensor shapes exceed the compiled tile size).
  * `allocation_size`: Total size (in Bytes) of the segment 7 allocation needed for this binary. The runtime allocates this space during JobPlan loading. The allocation is a single contiguous block that holds the program binary (at offset 0), and conditionally includes space for program correction tensors and intermediate data tensors.
  * `breakdown_jobbinary`: Size of job binary (in Bytes)
  * `breakdown_correctiondata`: Size of the data needed for program correction owing to symbolic start addresses and shapes (in Bytes)
  * `breakdown_tensordata`: Size of intermediate data tensors that spilled over to device memory (in Bytes)
  * `composite_address`: Set by the runtime after the binary is loaded to device (not specified by the compiler). A `CompositeAddress` identifying where the binary resides on device.
* `H2D`: Triggers a data transfer from host to Spyre. The runtime sends a control message to the card firmware to generate a DMAI control block to effect the transfer. Its attributes are:
  * `host_address`: Address of the tensor data on the host side. Set by the runtime at launch time (not specified by the compiler).
  * `device_address`: A `CompositeAddress` identifying where the tensor resides on the device. Set by the runtime at launch time (not specified by the compiler).
  * `size`: Size of the data transfer (in Bytes)
* `D2H`: Triggers a data transfer from Spyre to host. The runtime sends a control message to the card firmware to generate a DMAO control block to effect the transfer. Its attributes are:
  * `device_address`: A `CompositeAddress` identifying where the tensor resides on the device. Set by the runtime at launch time (not specified by the compiler).
  * `host_address`: Address of the tensor data on the host side. Set by the runtime at launch time (not specified by the compiler).
  * `size`: Size of the data transfer (in Bytes)

### Job Binary

The job binary is a binary file (not in text format) that contains all programs for the kernel. It is referenced by the `binary_path` attribute on the `ComputeOnDevice` step. The runtime (SpyreStream) is responsible for loading the binary to the Spyre device’s memory during JobPlan loading — it allocates a contiguous block in segment 7 via SpyreAllocator (using the `allocation_size` from the `ComputeOnDevice` step), transfers the binary via a DMA operation, and stores the resulting `CompositeAddress` on the step. The binary is placed at offset 0 within the allocated block.

### Program Correction Metadata

The program correction metadata is carried directly on the `ProgramCorrection` step (as the `program_correction_metadata` attribute). It contains information needed by the host function to interpret resolved symbol values and produce a correction tensor that can then be transferred to the device. An example of the use of program correction metadata in the context of kernel execution with symbolic start addresses and shapes is described in [example](#example2-job-plan-with-program-correction) below.

### Execution Flow Examples

#### Example1: Job plan for execution of a kernel with fixed tensor addresses and shapes

```
steps:
  1. ComputeOnDevice(binary_path="kernel.bin", expected_input_shapes=[[1024, 1024]],
       allocation_size=49152, breakdown_jobbinary=32768, breakdown_correctiondata=0, breakdown_tensordata=16384)
```

In this example, the compute kernel has tensors with fixed addresses and shapes. The job plan comprises a single `ComputeOnDevice` step. The `allocation_size` indicates the total amount of memory that needs to be reserved in segment 7 — in this example, 49152 bytes, broken down into 32768 bytes for the job binary (`breakdown_jobbinary`) and 16384 bytes for an intermediate data tensor (`breakdown_tensordata`). The runtime allocates this space during JobPlan loading, loads the binary to device, and stores the resulting `CompositeAddress` on the step. At launch time, the runtime constructs a compute control block and dispatches the binary.

#### Example2: Job plan for execution of a kernel with symbolic tensor addresses and shapes

This is a more complex example, wherein the compute kernel has tensors with symbolic start addresses and shapes. The symbol values are known only during kernel invocation and can change across consecutive launches of the same kernel. They are fed as input arguments when the kernel is invoked.

With symbolic tensor address/shapes, the job binary produced by the backend compiler cannot be executed as-is on the hardware. It needs to be edited just-in-time knowing the symbol values. This process is referred to as *program correction*. It is accomplished using the following job plan.

```
steps:
  1. ProgramCorrection(function=correct_fn, program_correction_metadata=hcm.json)
  2. H2D(host_address=<correction_tensor>, device_address=<seg7 offset>, size=2048)
  3. ComputeOnDevice(binary_path="kernel.bin", expected_input_shapes=[[1024, 1024]],
       allocation_size=51200, breakdown_jobbinary=32768, breakdown_correctiondata=2048, breakdown_tensordata=16384)
```

In this case, the job plan comprises 3 steps.
* The first step is `ProgramCorrection`, which performs a host-side computation. At launch time, the runtime (SpyreStream) provides the resolved symbol values (tensor addresses and shape values from the SpyreTensors' `CompositeAddress` values) to the host function, along with the `program_correction_metadata` (*hcm.json*). The metadata contains information pertaining to how the symbols must be interpreted in the context of the job binary. For example, if a shape of a dimension in a tensor is symbolic during compilation, then its value will be used to correct one of the loop counts in the job binary. The host function produces a correction tensor, which the runtime writes to a pre-allocated buffer (sized according to `breakdown_correctiondata` from the `ComputeOnDevice` step).
* The second step is `H2D`, which transfers the correction tensor to a reserved location on device within the segment 7 program allocation (at a compiler-specified offset after the program binary). The `host_address` and `device_address` are populated by the runtime at launch time.
* The third step is `ComputeOnDevice`, which launches the unified binary. The binary internally contains both a correction program and the compute program — the correction program reads the correction tensor to patch the compute program, then the compute program executes, successfully completing the kernel execution with the desired tensor address/shape. The `allocation_size` (51200 bytes) breaks down into 32768 bytes for the job binary, 2048 bytes for correction data, and 16384 bytes for intermediate tensor data.

## **Metrics **

## **Drawbacks**

## **Alternatives**

## **Prior Art**

## **How we teach this**

## **Unresolved questions**

## Resolution

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.

#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.

### Next Steps
Will implement it.

#### Tracking issue
https://github.com/torch-spyre/torch-spyre/issues/277

#### Exceptions
