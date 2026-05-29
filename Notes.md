# PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications

## Metadata

- Paper: PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications
- Authors: Kuntai Du, Bowen Wang, Chen Zhang, Yiming Cheng, Qing Lan, Hejian Sang, Yihua Cheng, Jiayi Yao, Xiaoxuan Liu, Yifan Qiao, Ion Stoica, Junchen Jiang
- Venue: SOSP 2025
- Topics: LLM inference engines, prefill-only workloads, KV cache management, scheduling, prefix caching

## Background and Problem

Most existing LLM inference engines are designed for generative workloads, such as chat, code generation, and summarization. These workloads usually consist of two phases: a prefill phase that processes the full input prompt, followed by multiple decode steps that generate output tokens one at a time. Because later decode steps need to attend to previous tokens, existing engines keep the KV cache for all tokens and all layers.

The paper observes a different and increasingly important class of workloads. LLMs are now being used in discriminative applications such as recommendation, credit verification, and data labeling. In these settings, the application often does not need a long generated answer. It only needs the probability distribution of one output token. For example, a recommendation system may ask whether a post should be recommended and use `P(Yes)` as the recommendation score.

The authors call this type of request a *prefill-only request* and the corresponding workload a *prefill-only workload*.

Prefill-only workloads have two key properties:

- Each request only needs one generated token, so it does not require multiple decode steps.
- Inputs are often long, because they may include user profiles, browsing histories, credit records, or documents.

Existing inference engines do not fully exploit these properties. They still preserve full KV caches as if the request may continue decoding, which wastes GPU memory for long inputs. They also rely on scheduling policies that are not aware of job completion time, even though prefill-only request durations are more predictable than generative requests.

## Main Idea

PrefillOnly is an inference engine specialized for prefill-only workloads. Its central observation is that, if a request does not need future decode steps, the system does not need to keep the full KV cache for the current request. This allows the engine to redesign memory management and scheduling around the actual workload.

The system has three main techniques:

1. **Hybrid Prefilling**
2. **Suffix KV Cache Discarding / Offloading**
3. **Shortest Prefill First with Continuous Prefill Time Estimation**

## Hybrid Prefilling

In a prefill-only request, the KV cache of a layer is usually no longer needed after that layer finishes computing attention. There will be no later decode token that comes back and attends to the historical K/V of that layer.

However, the paper shows that simply keeping only one layer of KV cache is not sufficient. The forward pass of an LLM also allocates large temporary tensors, especially inside MLP and linear layers. These intermediate tensors can be much larger than the KV cache of a single layer and can still dominate peak GPU memory usage.

PrefillOnly addresses this with hybrid prefilling:

- Attention layers are not chunked, so they can process the full context and preserve attention kernel efficiency.
- Non-attention layers, especially MLP and linear layers, are processed chunk by chunk to reduce peak memory from intermediate tensors.
- The implementation uses `torch.compile` to transform the computation graph. Consecutive linear operations are grouped into a virtual layer, which is then executed chunk by chunk.
- Output preallocation and in-place computation further reduce memory overhead.

The key benefit is that PrefillOnly reduces memory peaks from MLP intermediate tensors without degrading attention computation.

## Suffix KV Cache Discarding / Offloading

PrefillOnly must also preserve the benefits of prefix caching. In many prefill-only applications, multiple requests share a long prefix. For example, a recommendation system may evaluate many candidate posts for the same user, so all requests share the same user profile prefix.

Therefore, PrefillOnly does not simply discard all KV caches. Instead, it:

- keeps prefix KV caches as much as possible for future reuse;
- discards or offloads suffix KV caches when the request exceeds available GPU cache capacity;
- currently implements suffix KV cache discarding, while noting that systems such as LMCache or Mooncake Store could be used for offloading.

Hybrid prefilling enables this technique. Since attention layers are not chunked, the prefill can complete in one forward pass. The system can discard or offload suffix KV caches during execution without needing to reload or regenerate them for later chunks, which would be required by ordinary chunked prefilling.

## Scheduling: Shortest Prefill First with Continuous Estimation

Because prefill-only requests always generate one token, their completion time is more predictable than that of generative requests. PrefillOnly uses this property to schedule requests with a shortest-job-first-style policy, which the paper calls shortest prefill first.

The complication is prefix caching. A request's prefill time changes dynamically depending on whether its prefix KV cache is available. If the prefix hits in the cache, only the cache-miss suffix needs to be computed. If the cache has been evicted, the request becomes more expensive.

PrefillOnly therefore uses continuous prefill time estimation:

- before each scheduling decision, it re-estimates the prefill time of every waiting request;
- the estimate considers input length and the number of prefix-cache-hit tokens;
- the number of cache-miss tokens is used as a proxy for prefill time;
- a `-lambda * queue_time` term is added to reduce starvation of long-waiting requests.

This scheduling policy reduces latency while also improving prefix cache hit rate.

## Evaluation

The authors implement PrefillOnly on top of vLLM with about 4.6k lines of Python code. The evaluation covers:

- 4 hardware setups: NVIDIA L4, A100, H100 PCIe, and H100 with NVLink;
- 3 LLMs;
- 2 simulated workloads: post recommendation and credit verification;
- baselines including PagedAttention, chunked prefilling, pipeline parallelism, and tensor parallelism.

The main results are:

- PrefillOnly supports around 1.4 to 4.0 times higher QPS without increasing average latency or P99 latency.
- Compared with non-parallel baselines, PrefillOnly increases maximum input length by up to about 5 times.
- Hybrid prefilling significantly increases maximum input length without the throughput loss caused by chunked prefilling.
- PrefillOnly performs especially well under high QPS. Under low QPS, parallel baselines can sometimes provide lower single-request latency, but their communication and synchronization costs limit high-load throughput.

The evaluation also uses a specific routing strategy. For PrefillOnly and non-parallel baselines, the authors run one full inference engine instance on each GPU. They use user-id-based routing: requests from the same user are always routed to the same instance to maximize prefix cache reuse, while different users are assigned using round-robin routing for load balance.

## Limitations

The paper discusses several limitations:

- The current implementation discards suffix KV caches instead of fully implementing offloading, which may lose reuse opportunities for future requests.
- If the workload has heavy long-prefix sharing and prefix cache capacity becomes the main bottleneck, tensor parallelism or pipeline parallelism may be preferable because they distribute KV cache storage across multiple GPUs.
- PrefillOnly's scheduling is performed inside the inference engine. Large-scale deployments may benefit from global shortest-prefill-first routing at the router layer.
- The paper mainly focuses on throughput. There is still room for latency-centric optimizations.

## Questions and Clarifications During Reading

While reading the paper, my first confusion was about the basic workflow of LLM inference. I later understood ordinary generative inference as two phases. **Prefill** processes the full prompt in one forward pass, computes the K/V tensors for attention layers, and produces the probability distribution for the first output token. **Decode** then generates tokens one by one. Each decode step feeds only the newly generated token while reusing the historical KV cache. This distinction explains why a prefill-only workload is special: it only needs the probability distribution available after prefill and does not need the following decode loop.

I also clarified what KV cache actually stores. KV cache does not store training-time values or model parameters. It stores runtime K/V activations computed for the current inference request. The trainable parameters are projection matrices such as `W_Q`, `W_K`, and `W_V`; during inference, each token and each layer uses these matrices to compute Q/K/V from hidden states. The cache stores K and V because future decode tokens will produce their own Q and attend to historical K/V.

Another important question was why there are many attention layers and what "one layer of KV cache" means in PrefillOnly. Here, a layer refers to the attention layer inside a Transformer layer or Transformer block. In normal generative inference, all layers' KV caches must be retained because a new decode token passes through every Transformer layer, and at layer `l` it must attend to historical tokens' K/V from the same layer `l`. Since PrefillOnly does not continue decoding, once an attention layer finishes, that layer's K/V is usually no longer needed by the current request. This is why, conceptually, PrefillOnly only needs the currently active layer's K/V during execution.

I also clarified the relation between prefix caching and suffix KV cache discarding/offloading. Prefix caching means that when several requests share the same prompt prefix, the engine keeps the KV cache for that prefix so later requests can reuse it. For example, in recommendation, the same user profile may be shared by many candidate-post requests. PrefillOnly therefore cannot simply drop all KV caches. It keeps the more reusable prefix KV cache and discards or offloads the less reusable suffix KV cache.

The connection between Hybrid Prefilling and Suffix KV Cache Offloading became clearer as follows: Hybrid Prefilling is a change to the computation pattern, while suffix offloading/discarding is a cache-management policy. Hybrid Prefilling keeps attention layers unchunked and chunks non-attention layers, reducing peak memory from MLP intermediate tensors and allowing suffix KV caches to be discarded or offloaded during one complete prefill pass. Without hybrid prefilling, ordinary chunked prefilling may need the K/V from previous chunks again, making suffix cache discard/offload less efficient.

Finally, the routing setup in Section 7.1 helped me understand the evaluation. For non-parallel methods, the authors run one full inference engine instance per GPU and use user-id-based routing: requests from the same user go to the same instance to improve prefix cache hits, while different users are assigned round-robin for load balance. By contrast, tensor parallelism and pipeline parallelism run one model instance across multiple GPUs, so model parameters and KV caches are distributed across GPUs. This provides more effective prefix KV cache space, but introduces communication and synchronization overhead.

## Takeaways

The paper is important because it challenges the assumption that LLM serving systems should always be optimized for long-form generation. Many real applications use LLMs as discriminative models and only need one output token or one embedding-like representation. These workloads have different bottlenecks from chat-style generation.

PrefillOnly's design follows directly from the workload:

- no multi-step decoding means full per-request KV cache retention is unnecessary;
- long inputs require controlling both MLP intermediate tensors and KV cache memory;
- fixed output length enables job-completion-time-aware scheduling;
- repeated prefixes make it important to retain the most reusable prefix KV caches rather than dropping all caches.

Overall, PrefillOnly is a systems paper that identifies a new serving workload and revisits memory management, caching, and scheduling around that workload's actual requirements.

# MCM-GPU: Multi-Chip-Module GPUs for Continued Performance Scalability

## Metadata

- Paper: MCM-GPU: Multi-Chip-Module GPUs for Continued Performance Scalability
- Authors: Akhil Arunkumar, Evgeny Bolotin, Benjamin Cho, Ugljesa Milic, Eiman Ebrahimi, Oreste Villa, Aamer Jaleel, Carole-Jean Wu, David Nellans
- Venue: ISCA 2017
- Topics: GPU architecture, multi-chip modules, chiplet GPUs, NUMA, on-package interconnect, CTA scheduling, cache hierarchy

## Background and Problem

Historically, GPU performance scaling has relied heavily on transistor scaling. As process technology improved, a single GPU die could integrate more Streaming Multiprocessors (SMs), larger caches, and higher memory bandwidth. The paper argues that this path is becoming increasingly limited:

- Moore's Law is slowing down, so transistor density no longer improves at the historical rate.
- Photolithography reticle limits constrain the maximum manufacturable die size.
- Very large dies have poor yield and high cost.
- Even when applications have enough parallelism, a monolithic GPU cannot keep increasing SM count indefinitely.

Multi-GPU systems are a natural alternative, but they introduce their own limitations. Multiple complete GPUs are usually connected through board-level or system-level interconnects, which have lower bandwidth, higher latency, and higher energy cost. They also often require programmers or runtimes to handle work partitioning, data placement, synchronization, and load balancing.

The paper asks the following question: **Can a single logical GPU continue scaling without building an excessively large monolithic die?**

## Main Idea

The paper proposes *MCM-GPU*, a GPU organization that integrates multiple GPU Modules (GPMs) in the same package, connects them using on-package interconnects, and exposes the whole package as one logical GPU to the OS and programmers.

MCM-GPU sits between monolithic GPUs and multi-GPU systems:

- A monolithic GPU places all compute and memory-system resources on one large die.
- A multi-GPU system connects multiple complete GPUs at the board or system level.
- An MCM-GPU connects multiple smaller GPMs or chiplets inside one package and makes them behave like one GPU.

The main evaluated design is a 256-SM MCM-GPU built from four 64-SM GPMs. Each GPM contains part of the SMs, private L1 caches, a crossbar, an L2 slice, and a local DRAM partition. Together, the GPMs provide a unified address space and one logical GPU.

## MCM-GPU Organization

The organization of the L2 cache is easy to misunderstand. A precise description is:

> The L2 is logically shared but physically distributed and sliced.

The total L2 is composed of L2 slices located inside the GPMs. All SMs can access any L2 slice, but each physical address has exactly one home GPM, L2 slice, and DRAM partition.

For example:

```text
An SM in GPM0 accesses address A

If A maps to GPM0:
    GPM0 SM -> GPM0 L2 slice -> GPM0 DRAM

If A maps to GPM2:
    GPM0 SM -> inter-GPM link -> GPM2 L2 slice -> GPM2 DRAM
```

In the baseline MCM-GPU, each L2 slice caches only the data belonging to its local DRAM partition. A cache line has only one L2 location, so the L2 slices do not require a complex hardware coherence protocol.

## On-Package Bandwidth Considerations

This section asks: **How much bandwidth is needed between GPMs so that the interconnect does not bottleneck the MCM-GPU?**

The paper estimates this using a 4-GPM system. Let each GPM's local DRAM partition provide bandwidth `b`; the total DRAM bandwidth is therefore `4b`. In the evaluated design:

```text
Total DRAM bandwidth = 3 TB/s
Per-GPM local DRAM bandwidth = b = 768 GB/s
```

If addresses are finely interleaved across all memory partitions, a GPM's memory requests do not all go to its local partition. Many requests go to remote partitions. To fully utilize the total DRAM bandwidth, the inter-GPM bandwidth ideally needs to approach the aggregate DRAM bandwidth, about `3 TB/s` in this design.

However, such an interconnect would be costly in routing complexity, power, and package design effort. Based on NVIDIA's Ground-Referenced Signaling (GRS), the authors treat `768 GB/s` as a more realistic, lower-cost, lower-power design point. GRS supports about `20 Gb/s` per wire, so a `768 GB/s` link requires hundreds of parallel high-speed wires inside the package.

This motivates the rest of the paper: the baseline `768 GB/s` inter-GPM link is below the ideal bandwidth requirement, so the architecture needs mechanisms that reduce actual inter-GPM traffic.

## Three Locality Optimizations

The paper proposes three locality-aware mechanisms. Their shared goal is to reduce remote accesses and inter-GPM bandwidth pressure.

### Remote-Only L1.5 Cache

The L1.5 cache is a new GPM-side cache placed between the SM-private L1 caches and the remote L2/DRAM path. It primarily caches remote data previously accessed by the local GPM.

In the baseline design, if GPM0 accesses data homed in GPM2, later accesses may still need to cross the inter-GPM link. With L1.5, the first remote access can fill GPM0's L1.5 cache, allowing later accesses to hit locally and avoid inter-GPM traffic.

The paper evaluates six L1.5 configurations:

| Configuration | Meaning |
|---|---|
| `8 MB L1.5` | 8 MB total L1.5, caching both local and remote data |
| `8 MB Remote-Only L1.5` | 8 MB total L1.5, caching only remote data |
| `16 MB L1.5` | 16 MB total L1.5, caching both local and remote data |
| `16 MB Remote-Only L1.5` | 16 MB total L1.5, caching only remote data |
| `32 MB L1.5` | 32 MB total L1.5, caching both local and remote data, with extra cache capacity |
| `32 MB Remote-Only L1.5` | 32 MB total L1.5, caching only remote data, with extra cache capacity |

The 8 MB and 16 MB configurations are iso-transistor designs because they mostly move capacity from the original L2 to the L1.5 cache. The 32 MB configurations add extra cache capacity and are therefore not area-equivalent.

The experiments show that remote-only allocation better matches the purpose of L1.5. Local data is already served by the local L2, while remote data is more expensive and should be prioritized. When L1.5 is evaluated alone, `16 MB Remote-Only L1.5` is the best iso-transistor configuration.

### Distributed CTA Scheduling

CTA stands for Cooperative Thread Array and is essentially equivalent to a CUDA thread block. It is the unit scheduled onto GPU SMs.

The baseline centralized scheduler assigns CTAs globally based on SM availability, which can scatter neighboring CTAs across different GPMs:

```text
CTA0 -> GPM0
CTA1 -> GPM1
CTA2 -> GPM2
CTA3 -> GPM3
```

Neighboring CTAs often access neighboring or overlapping data. Scattering them across GPMs reduces locality and increases inter-GPM traffic.

Distributed CTA scheduling groups contiguous CTAs and assigns each group to the same GPM:

```text
CTA0, CTA1, CTA2, CTA3 -> GPM0
CTA4, CTA5, CTA6, CTA7 -> GPM1
```

This improves L1.5 cache reuse and reduces remote communication. The paper also notes that fixed coarse-grain grouping can introduce load imbalance, so a dynamic mechanism for choosing group size may provide additional benefit.

### First-Touch Page Placement

First-touch (FT) is a page placement policy. A memory page is mapped to the local memory partition of the GPM that first touches it.

For example:

```text
CTA-X on GPM0 first accesses page P0
=> P0 is placed in GPM0's memory partition

GPM1 later accesses P0
=> P0 remains in GPM0
=> GPM1 performs a remote access
```

Thus, FT is a placement policy, not a dynamic page migration mechanism. Once a page is placed, it generally stays there.

FT works especially well with distributed CTA scheduling. Many GPU applications repeatedly launch the same kernels, and CTAs with the same index across kernel invocations tend to access the same pages. If distributed scheduling keeps those CTAs on the same GPM, FT keeps their pages local to that GPM across kernel invocations.

## Evaluation

The paper evaluates the design using an NVIDIA in-house simulator. The modeled GPU resembles Pascal but is scaled to 256 SMs. The baseline configuration includes:

- 4 GPMs;
- 256 SMs in total;
- 128 KB L1 data cache per SM;
- 16 MB total L2 cache;
- a ring inter-GPM interconnect with 768 GB/s per link and 32 cycles per hop;
- 3 TB/s total DRAM bandwidth;
- 100 ns DRAM latency.

The benchmark suite contains 48 workloads from CORAL, Lonestar, Rodinia, and NVIDIA in-house CUDA workloads. The paper classifies them into high-parallelism and limited-parallelism workloads, and further separates high-parallelism workloads into memory-intensive and compute-intensive categories.

The main results are:

- The baseline MCM-GPU suffers from NUMA effects when inter-GPM bandwidth is below 3 TB/s, especially for memory-intensive workloads.
- Remote-only L1.5 alone improves average performance by about 5.2% and significantly reduces inter-GPM bandwidth.
- L1.5 plus distributed CTA scheduling improves memory-intensive workloads by about 23.4% on average.
- After adding FT, the best configuration becomes `8 MB Remote-Only L1.5 + 8 MB L2 + Distributed Scheduling + First Touch`, because FT makes many accesses local and therefore makes L2 capacity important again.
- Combining all three mechanisms improves performance by 22.8% over the baseline MCM-GPU and reduces inter-GPM bandwidth by 5x.
- The optimized MCM-GPU is 45.5% faster than the largest assumed manufacturable 128-SM monolithic GPU.
- It is 26.8% faster than an equivalently equipped multi-GPU system and comes within 10% of an ideal but unbuildable 256-SM monolithic GPU.

## Limitations

The paper has several important limitations:

- The results are based on an NVIDIA in-house simulator and forward-looking hardware assumptions, not measurements from a real chip.
- The `768 GB/s` inter-GPM bandwidth is estimated from GRS signaling capabilities, but the paper does not build a physical MCM-GPU.
- The inter-GPM topology is modeled at a relatively high level using ring or mesh assumptions, without a deep exploration of topology, routing, or congestion control.
- Distributed CTA scheduling uses fixed grouping and can create load imbalance.
- FT page placement does not include dynamic page migration, so pages can still be remote if the first toucher differs from later dominant accessors.
- The L1.5 cache is write-through to fit the GPU software coherence model, which can increase DRAM traffic for write-heavy workloads.

## Questions and Clarifications During Reading

My initial confusion was about the relation among chiplets, MCMs, and multi-GPU systems. I later clarified that a chiplet is a component, namely a smaller die; an MCM is a packaging method that integrates multiple dies or chiplets in one package; a multi-GPU system connects multiple complete GPUs at the board or system level. MCM-GPU aims to combine multiple GPMs into one logical GPU rather than expose multiple GPUs.

Another important concept is the package. In this paper, package does not mean a software package. It refers to the chip packaging level between an individual die and a PCB board. MCM-GPU uses package-level interconnects so that communication among GPMs is closer, wider, and more energy-efficient than board-level multi-GPU communication.

I also clarified the L2 cache organization. When Section 4 says shared L2, it means the L2 is logically shared by all SMs. Physically, however, the L2 is sliced and distributed across GPMs. Each address has one home L2 slice and DRAM partition. Therefore, the L2 is a distributed shared cache, not a centralized L2.

The L1.5 cache also needs to be distinguished from L2. L2 is a memory-side cache for the local DRAM partition of a GPM. L1.5 is a GPM-side cache that primarily keeps remote data previously accessed by that GPM. It is not a global shared cache, but a local cache for remote-data replicas.

For on-package bandwidth, my understanding is that the paper does not claim `768 GB/s` was already implemented in a product. Instead, it treats this bandwidth as a feasible design point based on GRS signaling trends. Ideally, inter-GPM bandwidth should approach the 3 TB/s aggregate DRAM bandwidth, but that would be expensive. The authors therefore choose 768 GB/s and rely on caching, CTA scheduling, and page placement to reduce actual cross-GPM traffic.

## Takeaways

The importance of this paper is that it systematically applies the chiplet/MCM idea to GPU performance scaling. Its key insight is not simply to place several small GPUs together. Rather, it recognizes that such a design naturally becomes a NUMA GPU and therefore requires locality-aware cache, scheduling, and page-placement mechanisms.

The MCM-GPU design follows a clear logic:

- use multiple smaller GPMs to avoid the manufacturing limits of a very large monolithic die;
- use package-level interconnects to avoid the high latency, high energy, and programming complexity of board-level multi-GPU systems;
- expose one logical GPU so that much of the complexity is hidden in hardware and drivers;
- use L1.5 caching, distributed CTA scheduling, and first-touch page placement to reduce NUMA penalties.

Overall, MCM-GPU is a classic architecture paper. It identifies a physical scaling limit, proposes a new package-level organization, and then designs mechanisms around the new bottleneck introduced by that organization. It is highly relevant to later work on chiplet GPUs and advanced-package GPU architectures.

# LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation

## Metadata

- Paper: LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation
- Authors: Yizhou Shan, Yutong Huang, Yilun Chen, Yiying Zhang
- Venue: OSDI 2018
- Topics: resource disaggregation, distributed OS, splitkernel, remote memory, RDMA, datacenter OS

## Background and Problem

Traditional datacenters use a monolithic server as the unit of deployment, operation, and failure. A server contains CPUs, DRAM, disks, NICs, power supply, motherboard, and other components. An application normally has to use CPU and memory from the same physical machine. The paper argues that this server-centric model has four major limitations.

First, resource utilization is inefficient. Since CPU and memory allocation is constrained by physical server boundaries, a cluster can still have idle CPU and idle memory while being unable to place a job that needs both in the same machine. The paper uses Google and Alibaba production traces to show that aggregate CPU and memory utilization are both around half.

Second, hardware elasticity is poor. Once devices are packaged into a server, it is hard to add, remove, move, or reconfigure only one type of resource. Datacenter operators must plan server configurations in advance, while application requirements change quickly.

Third, the failure domain is coarse. A failure in a critical server component, such as a CPU, memory controller, motherboard, or power supply, can make the whole server unusable even if other components still work.

Fourth, monolithic servers provide bad support for hardware heterogeneity. Modern datacenters increasingly use GPUs, TPUs, DPUs, FPGAs, NVM, and NVMe SSDs. In a server-centric model, these devices must fit into full server designs, including motherboards, power, cooling, drivers, and deployment workflows. This makes new hardware adoption slow and expensive. It can also force operators to buy complete servers just to obtain one specialized device, leaving other resources underutilized.

The paper's central claim is that datacenters should break servers into independent network-attached hardware components. Each component has its own controller and network interface, can scale independently, and can fail independently.

## Main Idea

LegoOS is not just Linux plus remote memory or remote storage. It proposes a new OS architecture called *splitkernel*.

The core idea of splitkernel is that when hardware resources are disaggregated, OS functionality should be disaggregated as well. Instead of running all OS subsystems inside one kernel on one server, LegoOS splits traditional OS functions into monitors. Each monitor runs on a hardware component and manages that component locally.

LegoOS uses three component types:

- `pComponent`: processor component. It runs user processes and the process monitor, and manages CPU cores and ExCache.
- `mComponent`: memory component. It runs the memory monitor, and manages virtual memory, physical memory, address mappings, and memory reads/writes.
- `sComponent`: storage component. It runs the storage monitor and manages persistent file storage.

The user-facing abstraction is a `vNode`. A vNode is similar to a virtual server: it has its own ID, virtual IP address, and storage mount point. Users know that their applications run on vNodes, but they do not know which physical pComponents, mComponents, or sComponents are used internally. This is what the paper means when it says the internal execution status is transparent to users.

## Splitkernel and Coherence

Splitkernel does not provide cache coherence across components. Here, *coherent* mainly refers to cache coherence: when multiple processors or components cache and access the same data, whether the system automatically guarantees that they observe a consistent, up-to-date value.

Ordinary multicore servers usually provide hardware cache coherence inside one machine. If one core writes a shared memory location, another core's stale cache line can be invalidated or updated. LegoOS targets a different setting where pComponents and mComponents communicate over a datacenter network. Maintaining global coherence across this network would require substantial traffic and complex protocols.

LegoOS therefore chooses a non-coherent design:

- Components can still use coherence already provided inside one component.
- LegoOS does not maintain transparent shared-memory coherence across components.
- Cross-component communication uses explicit message passing.
- LegoOS does not support writable shared memory across processors.

This creates an important programming constraint. Threads within one process, which share writable memory, are scheduled on the same pComponent. Different processes should communicate through sockets, RPC, or application-level message passing instead of relying on writable shared memory across pComponents.

## pComponent, Virtual Cache, and ExCache

The most important design challenge is separating processors from memory. Traditional CPUs and OSes assume that page tables, TLBs, memory controllers, and main memory are local. LegoOS moves large-capacity memory and address mapping to mComponents.

To avoid doing virtual-to-physical address translation at pComponents, LegoOS organizes pComponent-side L1/L2/L3 caches as virtual caches, meaning virtually indexed and virtually tagged caches.

Virtual caches have two classic problems.

**Synonym problem** means one physical address can be mapped by multiple virtual addresses, which can create multiple cache copies of the same data. LegoOS avoids this by not supporting writable shared memory across processes.

**Homonym problem** means different address spaces can use the same virtual address for different data. For example, process A and process B can both use virtual address `0x400000`. If the cache only compares virtual addresses, B may incorrectly hit A's cache line. LegoOS solves this by storing an ASID, or address space ID, in each cache line. A cache hit is identified by `(ASID, virtual address)`, not by virtual address alone.

Since application memory is remote, accessing mComponents on every memory access would be too slow. LegoOS keeps a small amount of DRAM or HBM at each pComponent as `ExCache`. ExCache can be understood as an extra cache level after the traditional LLC:

```text
L1/L2/L3 virtual cache -> ExCache -> network -> mComponent DRAM/NVM
```

However, ExCache is not a normal LLC:

- It is backed by pComponent-side DRAM/HBM, not on-chip SRAM.
- It is a virtual cache.
- Its hit path is intended to be handled by hardware.
- Its miss path is handled by the LegoOS process monitor in software.
- Its main purpose is to hide remote-memory network latency.

The local DRAM/HBM at a pComponent is also partially reserved for LegoOS kernel data. This does not mean the kernel uses ExCache as ordinary application-data cache. Instead, the kernel needs local physical memory for its own data structures: thread state, scheduling queues, kernel stacks, RPC buffers, ExCache tags, FIFO/LRU metadata, cached vRegion arrays, and Linux syscall compatibility state. At boot time, LegoOS reserves a contiguous physical region and partitions it into ExCache, ExCache metadata, and kernel physical memory. The kernel accesses this local memory directly through physical addresses, so it does not need to fetch its own metadata from mComponents.

With this design, pComponents do not need traditional address mappings or TLB accesses. On an application memory access, the pComponent can use the virtual address to locate the ExCache set. If ExCache hits, there is no VA-to-PA translation and no TLB miss. Address mappings and physical memory allocation are handled by mComponents.

## Memory Management and vRegion

LegoOS allows a process address space to span multiple mComponents. The key question is: when a pComponent accesses a virtual address, how does it know which mComponent should handle the request?

`vRegion` solves this routing problem. LegoOS divides each process virtual address space into large fixed-size regions, for example 1 GB each. Every active vRegion has an owner mComponent. The owner handles memory accesses and virtual memory management for that address range.

For example:

```text
0 GB - 1 GB   vRegion 0 -> mComponent 1
1 GB - 2 GB   vRegion 1 -> mComponent 2
2 GB - 3 GB   vRegion 2 -> mComponent 2
3 GB - 4 GB   vRegion 3 -> mComponent 5
```

If an application accesses a virtual address around `1.3 GB`, the pComponent uses its cached vRegion array to find that the address belongs to vRegion 1 and sends the request directly to mComponent 2.

The paper calls this two-level distributed virtual memory management.

At the first level, the home mComponent makes coarse-grained decisions. Each process has one home mComponent. It stores the vRegion array, records which mComponent owns each vRegion, and handles high-level virtual memory operations such as `brk`, `mmap`, `munmap`, and `mremap`.

At the second level, each vRegion owner performs fine-grained VMA management. It maintains the vma tree for the address ranges it owns, records permissions, manages physical memory allocation, and performs virtual-to-physical address mapping locally.

This design has three benefits:

- pComponents can route memory accesses directly through cached vRegion arrays instead of contacting the home mComponent on every access.
- One process can use memory from multiple mComponents, improving capacity utilization and parallelism.
- The home mComponent only makes coarse-grained decisions, while owner mComponents handle local details, reducing central bottlenecks.

## Optimization on Memory Accesses

Section 4.4.2 optimizes ExCache cold misses. In the naive design, every ExCache miss goes to an mComponent. But many real applications frequently first-touch newly allocated anonymous memory, such as heap, stack, or anonymous mmap regions.

Anonymous memory is process memory that is not backed by a file. Examples include `malloc`, `brk`, stack growth, and `mmap(..., MAP_ANONYMOUS, ...)`. A key semantic property is that newly allocated anonymous memory initially reads as zero. The OS must not leak stale data from other processes, so new pages are logically zero-filled.

LegoOS uses this property to optimize first access:

```text
application allocates anonymous memory
-> pComponent records the address range and permissions
-> application first accesses the address
-> ExCache miss
-> LegoOS does not contact mComponent
-> LegoOS allocates a zero-filled line in ExCache
```

The paper calls such a line a `p-local line`. It exists only in the pComponent ExCache. The mComponent has not allocated physical memory or installed an address mapping for it yet. Only when the line becomes dirty and is evicted from ExCache does the pComponent send it to the owner mComponent. The mComponent then allocates physical memory, creates the mapping, and stores the data.

This is essentially delayed physical memory allocation until writeback time. The optimization avoids many remote memory accesses for first touches of anonymous memory. It is safe because LegoOS does not support writable shared memory across pComponents, so no other pComponent can concurrently read this p-local line.

## Storage Management and Stateless Design

LegoOS provides a POSIX-style hierarchical file interface. Users can read and write files under a vNode mount point. Internally, however, the storage monitor is simplified.

`Stateless storage server design` means that the sComponent avoids storing client session state or process-specific state. Every I/O request carries all information needed to execute the operation, such as vNode ID, full path name, file offset, size, and operation type. The sComponent does not need to know which process previously opened which file or what file descriptor points to which object.

In traditional Linux, `read(fd, ...)` depends on kernel-maintained state such as the fd table, open file object, inode, current offset, and permissions. LegoOS keeps these Linux-like states mostly in the syscall compatibility layer at the pComponent. The pComponent translates `fd` into full path, offset, and size before sending the request onward.

The common file access path is:

```text
pComponent -> mComponent -> sComponent
```

This is because LegoOS places the storage buffer cache at mComponents, not at sComponents. Storage controllers have limited memory, while mComponents are memory resources and are better suited for file caching.

On a read cache hit:

```text
pComponent -> mComponent -> pComponent
```

On a read cache miss:

```text
pComponent -> mComponent -> sComponent -> mComponent -> pComponent
```

Writes are similar: data is first written to the buffer cache managed by an mComponent, and then reaches sComponent on flush or fsync. Importantly, an sComponent does not require a dedicated mComponent. mComponents and sComponents are separate resource pools. For simplicity and to avoid coherence traffic, LegoOS currently places the buffer cache of one file on one mComponent.

## Global Resource Management and Failure Handling

LegoOS uses two-level resource management. At the higher level, it has three global managers:

- `GPM`: global process manager. It chooses the pComponent for a new process.
- `GMM`: global memory manager. It chooses the mComponent for a new vRegion.
- `GSM`: global storage manager. It manages storage resources.

These global managers make only coarse-grained decisions and maintain approximate resource usage information. Fine-grained decisions are made locally by component monitors. For example, a pComponent schedules threads locally, and an mComponent allocates virtual and physical memory locally. This keeps global managers out of common critical paths.

For reliability, the current LegoOS prototype mainly handles mComponent failures. Since one process's memory can span multiple mComponents, not handling memory component failures would increase application failure probability. LegoOS does not fully handle pComponent failure, and storage reliability is largely left to applications or future work.

LegoOS does not simply duplicate all memory because full replication would require at least 2x memory space. Instead, each vma uses a primary mComponent, a secondary mComponent, and a backup log file on an sComponent. The primary stores full memory data and metadata. The secondary stores a small append-only log and a copy of the vma tree. The log is flushed to the sComponent in the background. If the primary fails, LegoOS can attempt recovery by replaying logs from the secondary and sComponent.

## Implementation

LegoOS is implemented in C on x86-64. The prototype has about 206K SLOC, including 56K SLOC for drivers. It supports 113 Linux syscalls, 15 pseudo-files, and 10 vectored syscall opcodes. Supporting Linux ABI compatibility is important because it allows many datacenter applications to run with little or no modification.

Because real disaggregated hardware was unavailable, the paper emulates components using commodity servers. For example, it limits usable cores to emulate mComponent controllers, and limits local memory to emulate pComponents with software-managed ExCache.

Internal communication uses a custom RDMA-based RPC framework. ExCache misses, dirty-line writebacks, mComponent requests, and sComponent requests all rely on this low-latency network stack. LegoOS also implements socket-over-RDMA and a traditional TCP/IP socket stack for applications.

## Evaluation

The evaluation uses 10 machines. Each machine has two Intel Xeon E5-2620 2.40GHz processors, 128 GB DRAM, and a 40 Gbps Mellanox ConnectX-3 InfiniBand NIC. The machines are connected by a 40 Gbps InfiniBand switch. The Linux baseline is version 4.9.47.

Figure 8 evaluates remote memory throughput at mComponents. The experiment uses a large empty ExCache so that every 4 KB memory load becomes an ExCache cold miss and goes to an mComponent. The x-axis is the number of workload threads, and the y-axis is KIOPS. A legend such as `1M x 4-worker` means one mComponent with four worker threads. Here `M` means mComponent, not MB.

The main conclusions from Figure 8 are:

- More mComponent workers generally improve remote memory request throughput.
- More workload threads increase request concurrency and can better saturate mComponents.
- More mComponents can spread request load, but the benefit diminishes once the total worker count reaches around four.
- The `p-local` optimization significantly improves first-touch anonymous-memory performance because it bypasses mComponents.
- `p-local` is still slower than local Linux memory access because it still traps into the LegoOS kernel and sets up ExCache metadata.

Figure 10 evaluates PARSEC workloads including BlackScholes, Freqmine, and StreamCluster. LegoOS uses 128 MB ExCache, one mComponent, one worker, and one sComponent. The comparison is against a monolithic Linux server with enough memory. This figure shows LegoOS slowdown relative to an ideal local-memory Linux setup; it is not a systematic sweep of ExCache size.

Figures 11 and 12 evaluate TensorFlow and Phoenix. The x-axis is labeled `ExCache/Memory Size` because the paper sets LegoOS ExCache size equal to the local main memory size of Linux-swap baselines for fairness. For LegoOS, 128/256/512 MB refers to pComponent ExCache. For Linux-swap, ramdisk-swap, and InfiniSwap, it refers to local main memory, with the remaining working set handled by swap or remote memory.

This does not mean ExCache is semantically the same as main memory:

```text
LegoOS:
  ExCache = local cache
  mComponent = true large-capacity application memory

Linux-swap:
  local DRAM = local main memory
  SSD/ramdisk/remote memory = swap backing store
```

The comparison asks: if the processor has the same amount of fast local memory nearby, how do different systems handle the rest of the working set?

The main application results are:

- With ExCache set to around 25% of the working set, LegoOS is about 1.68x slower for TensorFlow and 1.34x slower for Phoenix compared with a memory-rich monolithic Linux server.
- LegoOS is usually significantly better than SSD swap and remote swap.
- Its performance comes from an efficient RDMA RPC stack, a shorter path than Linux paging, and the anonymous-memory first-access optimization.
- ExCache management matters. FIFO plus free list plus piggybacking dirty flush and miss fill gives the best performance. LRU performs worse due to lock contention.
- More mComponents do not always improve application performance, because dirty-line flushes and miss fills may go to different mComponents and therefore lose the piggyback optimization.
- Memory replication adds about 2% to 23% overhead.

The failure analysis estimates reliability using device MTTF values. Because LegoOS components are simpler and resource utilization can be higher, the paper estimates that LegoOS improves MTTF by about 17% to 49% over an equivalent monolithic server cluster. This is an analytical estimate, not a long-running hardware measurement, and it does not include transient failures.

## Limitations

The paper has several clear limitations:

- It does not use real disaggregated hardware; components are emulated with commodity servers.
- The pComponent virtual-cache and ExCache design depends on future or specialized hardware support.
- LegoOS does not support writable shared memory across pComponents, which constrains the application model.
- The current prototype mainly handles mComponent failure and does not fully handle pComponent or sComponent failure.
- The storage monitor is simplified and is not the main focus of the paper.
- Resource management and load balancing are still preliminary, and LegoOS does not support memory data migration.
- Performance heavily depends on memory-access locality, ExCache hit rate, RDMA network performance, and mComponent worker parallelism.

## Questions and Clarifications During Reading

My first question was about the phrase bad support for heterogeneity in Section 2.1. I later understood it as follows: traditional servers tightly bind hardware devices into one full-machine form factor. This makes GPUs, TPUs, FPGAs, NVM, and other new devices hard to add independently. A datacenter may need to buy an entire server just to get one type of accelerator, wasting other resources and slowing hardware adoption. LegoOS aims to make hardware types scale and enter the datacenter independently.

Another key concept was coherent. In this paper, it mainly means cross-component cache coherence. Ordinary multicore servers provide cache coherence among cores inside one machine. LegoOS does not provide such global coherence between pComponents and mComponents. It chooses explicit message passing to avoid the traffic and complexity of maintaining shared-memory coherence over a datacenter network.

I also clarified the vNode abstraction. Users see a virtual server-like vNode, not a set of physical components. One vNode can internally use multiple pComponents, mComponents, and sComponents. One physical component can also serve multiple vNodes. This hides internal execution placement from users.

For the homonym problem, my understanding is that a virtual cache can confuse identical virtual addresses from different processes. LegoOS uses ASID plus virtual address as the cache-line identity, so process A's `0x400000` cannot be incorrectly hit by process B's `0x400000`.

For pComponent-side DRAM/HBM, I initially mixed kernel physical memory with ExCache. The clearer view is that local memory is partitioned into ExCache, ExCache metadata, and kernel physical memory. ExCache caches remote application memory. Kernel physical memory stores LegoOS's own kernel state. ExCache metadata stores tags, permission bits, valid/dirty state, and replacement queues. The kernel manages ExCache, but it does not use ExCache as its main storage for kernel data.

For ExCache, my understanding is that it resembles an L4 cache behind the LLC, but it should not be treated as a normal LLC. It is a virtual, inclusive, software-managed extended cache backed by pComponent-side DRAM/HBM, mainly used to cache remote memory from mComponents. ExCache hits avoid network access. ExCache misses trap into LegoOS and are handled by the process monitor.

For vRegion, I understood it as a coarse-grained routing table from process virtual address ranges to mComponents. It solves the problem of finding the target mComponent when one process address space spans multiple mComponents. The home mComponent stores the vRegion array, while owner mComponents store vma trees and address mappings for their own vRegions.

For Optimization on Memory Accesses, my understanding is that LegoOS uses the zero-initialization semantics of anonymous memory. The first access to anonymous memory can create a zero-filled line in local ExCache instead of fetching from mComponent. Only when this p-local line becomes dirty and is evicted does the mComponent allocate physical memory and store the data.

For storage, my understanding is that file access usually follows `pComponent -> mComponent -> sComponent`. The mComponent holds the buffer cache, while the sComponent holds persistent data. A cache hit does not access the sComponent; a cache miss or fsync does. An sComponent does not require a dedicated mComponent, because the two are separate resource pools.

Finally, Figures 8, 10, and 11/12 answer different questions. Figure 8 measures remote-memory miss-path throughput. Figure 10 measures PARSEC slowdown against a memory-rich monolithic Linux server. Figures 11/12 set LegoOS ExCache size equal to Linux local memory size to fairly compare systems with the same amount of fast memory near the processor. This does not mean ExCache is semantically equivalent to main memory.

## Takeaways

The importance of this paper is that it turns hardware resource disaggregation into an OS architecture problem. Treating remote memory as swap or as an RDMA memory pool still preserves many assumptions of monolithic servers. LegoOS instead asks: if CPUs, memory, and storage become independent network-attached components, how should the OS itself be organized?

LegoOS follows a clear design logic:

- use splitkernel to place OS functionality close to hardware components and reduce remote management traffic;
- use vNodes to preserve a simple virtual-server abstraction for users;
- use non-coherent message passing instead of maintaining shared-memory coherence across the datacenter network;
- use virtual caches and ExCache so pComponents can run applications without local large-capacity memory or local TLB/address mapping;
- use vRegions to make distributed virtual address spaces routable and scalable;
- use mComponent-side buffer cache and stateless sComponents to simplify storage components.

Overall, LegoOS is a systems prototype paper. Its main contribution is not proving that resource disaggregation already outperforms conventional servers on current hardware. Instead, it opens a design space for an OS in a component-centric datacenter. Mechanisms such as ExCache, vRegion, p-local lines, and RDMA RPC all serve two goals: reducing cross-network accesses and avoiding centralized management.

The paper also leaves important open questions. Real hardware support for virtual cache and remote memory is still needed. Full cross-component failure handling is incomplete. The lack of writable shared memory limits some applications. Large-scale resource managers need better load balancing and memory migration. These limitations make LegoOS more of a direction-setting prototype than a drop-in replacement for Linux server clusters.

# Jenga: Effective Memory Management for Serving LLM with Heterogeneity

## Metadata

- Paper: Jenga: Effective Memory Management for Serving LLM with Heterogeneity
- Authors: Chen Zhang, Kuntai Du, Shu Liu, Woosuk Kwon, Xiangxi Mo, Yufeng Wang, Xiaoxuan Liu, Kaichao You, Zhuohan Li, Mingsheng Long, Jidong Zhai, Joseph Gonzalez, Ion Stoica
- Venue: SOSP 2025
- Topics: LLM serving, KV cache management, heterogeneous attention, prefix caching, GPU memory allocation

## Background and Problem

Existing LLM serving systems usually rely on PagedAttention-style memory management. PagedAttention works well for early Transformer models because those models are largely homogeneous:

- every layer has the same type of full attention;
- every token produces KV cache entries of the same size;
- each layer needs a similar number of cache pages;
- cache pages can therefore be partitioned by layer with a fixed page size.

Recent LLMs break these assumptions. Models now combine full attention, sliding-window attention, local attention, Mamba/state-space layers, cross-attention for vision inputs, and even multiple models in speculative decoding. These mechanisms create two forms of heterogeneity.

First, different tokens or layers may have different cache sizes. For example, a vision-language model may store text-token KV cache and image-token embeddings with different sizes. A speculative decoding engine may keep KV caches for both a small draft model and a large target model.

Second, different layers depend on different subsets of historical tokens. Full attention needs the whole prefix. Sliding-window attention only needs recent tokens within the window. Mamba-like layers may only need a compressed state, often associated with the latest token or periodic checkpoints. Therefore, memory demand changes with request length and layer type.

Figure 2b illustrates this point. In a heterogeneous model, the memory ratio of full attention, sliding-window attention, and Mamba is not fixed. Mamba may dominate short requests because its per-sequence state is large but almost independent of sequence length. Full attention dominates long requests because its KV cache grows linearly with the number of tokens. Sliding-window attention grows at first but saturates after the window size.

## Main Idea

Jenga is a layer-property-aware memory manager for heterogeneous LLM serving. Its central idea is to make the memory manager aware of each layer type's cache size and cache dependency pattern.

Jenga introduces a unified `LayerProperty` interface:

```python
class LayerProperty:
    def page_size()
    def active_pages(request, length)
    def possible_prefix(is_hit)
```

The three methods mean:

- `page_size`: the cache memory footprint per token or state for this layer type;
- `active_pages`: the pages that are actually needed for future token generation;
- `possible_prefix`: the prefix lengths that can be treated as cache hits under this layer's dependency rule.

Based on this abstraction, Jenga has three main components:

1. a global LCM allocator for heterogeneous page sizes;
2. layer-specific allocators that allocate and free only the active pages;
3. layer-specific prefix cache managers that customize hit and eviction policies.

## LCM-Based Memory Allocation

The allocation problem is that fixed-size pages either waste space or prevent memory from being exchanged among different layer types. A max-page strategy uses the largest page size for all layer types, which wastes memory for smaller cache objects. A static partition strategy splits memory into separate pools for each layer type, but it cannot adapt when request length or token-type distribution changes at runtime.

Jenga solves this with a two-level LCM allocator:

- the global allocator divides GPU KV-cache memory into large pages whose size is the least common multiple of all layer page sizes;
- each layer-specific allocator obtains large pages from the global allocator and subdivides them into small pages of its own type.

For example, if image-token pages are 256 bytes and text-token pages are 384 bytes, Jenga uses a 768-byte large page. The image allocator can split one large page into three 256-byte pages, while the text allocator can split one large page into two 384-byte pages.

This design has two benefits:

- large pages can be exchanged dynamically across layer types, reducing external fragmentation;
- attention kernels can still operate with per-layer page IDs and page sizes, so Jenga can reuse PagedAttention kernels with minimal changes.

## Memory Layout

PagedAttention traditionally uses a layer-page layout: first partition memory by layer, then partition each layer's region into pages. This is convenient for homogeneous Transformers because each layer has the same cache shape and usually needs the same number of pages.

This layout is not suitable for Jenga because different layer types can have different numbers of layers and different memory needs. For example, a model may have 10 full-attention layers and 52 sliding-window layers. If memory is fixed per layer, free memory in sliding-window layers cannot easily be reused by full-attention layers when long requests make full attention the bottleneck.

Jenga instead uses a page-layer layout. It first manages pages globally, then maps each page to the layer/type-specific view needed by the attention kernels. This allows memory to flow dynamically among full attention, sliding-window attention, Mamba states, and vision caches.

## Prefix Caching

Prefix caching lets a new request reuse the KV cache of an earlier request with a common prefix. In heterogeneous models, this becomes more complex for two reasons.

First, the model-level cache hit length is limited by the layer with the shortest hit length. If any layer misses a token, that token has to be recomputed through all layers. Therefore, caching must be balanced across layers.

Second, different attention mechanisms need different numbers of cached tokens to reach the same effective prefix hit length. Full attention needs all prefix tokens. Sliding-window attention only needs window-local tokens. Mamba may only need a state checkpoint. This means the optimal memory partition depends on request length, attention type, and workload reuse pattern.

Jenga uses all memory not allocated to running requests as an opportunistic cached page pool. When running requests need memory, cached pages can be evicted. Thus, the normal cached page pool does not use a fixed batch-cache partition.

However, Jenga adds a common page pool for pages that are predicted to be highly reusable. This pool is protected from ordinary eviction, so it explicitly trades off prefix-cache hit rate and batch size. Jenga estimates the best common-pool size by simulating different sizes and choosing the one with the lowest estimated total execution time.

## Customized Eviction and Hit Policies

Jenga customizes LRU eviction by updating last-access time only for active pages. Figure 11 explains this design.

For full attention, all prefix tokens are active because the next token may attend to all previous tokens. Therefore, their last-access times are updated whenever they are used.

For sliding-window attention, tokens outside the current window are not active. Even if they belong to a request prefix, they should not be treated as recently useful for this layer. Jenga therefore does not update their last-access times. They become older under LRU and are more likely to be evicted.

This is the key point of Figure 11: "recently used" should be defined according to each layer's actual attention dependency, not according to the whole request prefix.

Jenga also customizes cache-hit rules. For full attention, a prefix hit requires a continuous prefix from the beginning. For sliding-window attention, a prefix can be valid if the required window pages are present. For Mamba, Jenga caches periodic states, such as every 512 tokens, so only those checkpoint-aligned prefix lengths can hit.

## Customization for Different Layers

Section 6.3 shows how the layer-property abstraction is instantiated:

- Full attention: active pages are all prefix pages; valid hits require a complete continuous prefix.
- Sliding-window attention: active pages are only pages inside the sliding window; a prefix hit only requires the window-relevant pages.
- Mamba: only selected state checkpoints are cached, such as every 512 tokens; this reduces memory but makes prefix hits coarser-grained.
- Local attention: active pages are pages in the current local chunk; hit rules are chunk-local.
- Vision embedding and vision cross-attention cache: image-token caches should be evicted at image granularity, because evicting a single image token may force recomputation of the whole vision encoder output.

## Evaluation

Jenga is implemented in vLLM with about 4,000 lines of Python code. The evaluation covers heterogeneous models and use cases including Llama 3.2 Vision, Gemma-3, Ministral, Llama 4, Jamba, a Character.ai-style model, PyramidKV, and speculative decoding.

The main results are:

- On H100, Jenga improves throughput by up to 1.73x and 1.46x on average over vLLM.
- On L4, Jenga improves throughput by up to 2.16x and 1.65x on average.
- For homogeneous Llama 3.1, Jenga has almost no overhead.
- For Llama 4, Jenga supports much longer context lengths by freeing inactive pages. On 8 H200 GPUs, it increases the maximum supported context from 3.7M tokens to 14.7M tokens.
- For VLMs with chunked prefill, Jenga avoids recomputing the vision encoder and improves throughput significantly.
- For speculative decoding, Jenga automatically handles different KV-cache sizes for draft and target models.

## Limitations

Jenga's effectiveness depends on correctly defining layer properties for each new attention or state mechanism. Existing mechanisms can be parsed automatically, but genuinely new architectures still require corresponding property definitions.

The common page pool relies on workload reuse patterns and prediction. It works well when common prefixes recur within a short time window, but its benefit may shrink for workloads with little prefix sharing.

Jenga also focuses on GPU memory management inside an inference engine. It does not fully solve multi-node routing, distributed cache placement, or external KV-cache offloading.

## Questions and Clarifications During Reading

My first question was about Figure 2b. I later understood it as showing that memory ratios across layer types are request-length dependent. Mamba can dominate short requests because its state is large and roughly constant. Full attention dominates long requests because its KV cache grows with sequence length. Sliding-window attention saturates after the window size.

I also clarified why prefix caching is harder for heterogeneous attention. The model's effective hit length is bounded by the worst layer. If one layer misses after token 3 and another can hit token 5, the model can only reuse the first 3 tokens because recomputing token 4 requires all layers. Thus, the cache manager must balance hit lengths across layers, even though each layer type needs a different amount of memory to achieve the same hit length.

The sentence "this memory layout is not suitable for Jenga, because different layer types can have different number of layers" refers to the limitation of layer-first memory partitioning. If memory is pre-partitioned per layer, then memory owned by one type of layer cannot flexibly serve another type. Jenga needs page-level exchange across layer types.

I also clarified the relation between batch size and prefix cache size. Jenga's normal cached page pool is opportunistic: running requests have priority, and leftover memory becomes cache. The common page pool is the part that explicitly protects high-value prefix pages and therefore introduces a controlled trade-off between larger batch size and higher prefix hit rate.

Figure 11 became clearer after interpreting last-access time as layer-specific. For full attention, all prefix tokens are truly active. For sliding-window attention, old tokens outside the window are not active and should not be refreshed in LRU. This lets Jenga evict useless pages earlier.

Finally, Section 6.3 is best understood as examples of the same `LayerProperty` interface. Full attention, sliding window, Mamba, local attention, and vision caches all have different active-page and possible-prefix definitions, but they can be managed by the same framework.

## Takeaways

Jenga is important because it updates a core assumption in LLM serving systems. PagedAttention assumes a relatively homogeneous Transformer. Modern LLMs are increasingly heterogeneous, so memory management must understand layer-specific cache sizes and dependency patterns.

The paper's design logic is direct:

- heterogeneous cache sizes require a fragmentation-resistant allocator;
- heterogeneous attention dependencies require layer-specific active-page tracking;
- prefix caching requires hit and eviction policies that match each layer's dependency rule;
- batch size and cache hit rate must be traded off carefully when cached pages compete with running requests.

Overall, Jenga is a systems paper about making the memory manager aware of model architecture. It does not propose a new attention mechanism. Instead, it provides a general memory-management framework that lets inference engines serve modern heterogeneous models with higher GPU memory utilization and better throughput.

# Rearchitecting the Thread Model of In-Memory Key-Value Stores with μTPS

## Metadata

- Paper: Rearchitecting the Thread Model of In-Memory Key-Value Stores with μTPS
- Authors: Youmin Chen, Jiwu Shu, Yanyan Shen, Linpeng Huang, Hong Mei
- Venue: SOSP 2025
- Topics: in-memory key-value stores, thread architecture, RDMA, cache efficiency, contention mitigation, auto-tuning

## Background and Problem

Modern in-memory key-value stores often run in user space and use kernel-bypass networking stacks such as RDMA, DPDK, or libibverbs. Server worker threads are pinned to CPU cores, busy-poll network queues, receive client `GET`, `PUT`, and `DELETE` requests, and access in-memory indexes and value storage. With modern NICs and memory systems, a single KVS operation may have only a few hundred nanoseconds to a few microseconds of processing budget.

Most high-performance KVSs use a run-to-completion non-preemptive thread-per-queue (NP-TPQ) model. A worker thread handles a request from beginning to end:

```text
poll network buffer -> parse request -> index lookup -> copy/update value -> send response
```

This model avoids OS scheduling and context-switch overhead, but it mixes sub-tasks with very different memory access patterns. Network buffers and hot items can often remain cache-resident, while index traversal and data access touch a much larger memory space and cause many cache misses. When these steps run inside the same monolithic function, they can pollute each other's cache working sets.

The paper asks whether the conventional run-to-completion thread model is still appropriate for KVSs operating at tens of millions of operations per second.

## Main Idea

μTPS reintroduces the thread-per-stage (TPS) idea, but it does not split the system into many functional stages. Instead, it splits request processing by cache residency:

- **Cache-resident layer (CR layer)** handles network buffers, request parsing, response sending, and some hot items. Its goal is to keep frequently accessed data in CPU caches.
- **Memory-resident layer (MR layer)** handles the full index and full data items. Its goal is to use batching, prefetching, and coroutines to amortize cache-miss costs.

Thus, μTPS is different from traditional TPS. Traditional TPS often uses DRAM queues and buffering to hide slow disk or network I/O. μTPS uses stage separation to isolate cache-friendly work from cache-unfriendly work and to reduce cache pollution and write contention.

## Why CAT Helps DDIO Only Modestly

The paper first examines a natural alternative: since Intel DDIO places NIC-received data into a small number of LLC ways, perhaps Intel Cache Allocation Technology (CAT) can isolate those DDIO-reserved ways and protect network buffers from CPU pollution.

The improvement is small because DDIO-reserved ways are not a strict network-buffer isolation region. DDIO typically allocates into reserved ways on cache misses, but if a cache line is already present in other LLC ways, DDIO can access or update that line there.

In the run-to-completion model, worker threads actively load network buffers while polling and parsing requests. These loads can bring network-buffer cache lines into ordinary LLC ways available to the worker. The same worker then performs index traversal and data access, whose random memory accesses can evict those network-buffer cache lines. CAT can partly partition cache space, but it cannot solve the deeper problem that one execution stream mixes network processing and memory-resident index/data access.

μTPS addresses the problem by separating execution streams. CR threads mostly touch network buffers and hot items, while MR threads touch the full index and data. As a result, MR-layer cache misses are less likely to continuously evict the CR layer's cache working set.

## μTPS Design

### Reconfigurable RPC

Traditional RDMA/RPC systems often bind client connections or request routing to server worker threads. If the number of workers changes, clients may need to learn the new routing information. This is too expensive for μTPS because the CR layer's worker count changes dynamically.

μTPS introduces reconfigurable RPC based on a shared receive buffer and Shared Receive Queue (SRQ). Clients send requests to the server-side shared queue without knowing how many workers are active. Server workers divide receive-buffer slots by modulo:

```text
slot m -> worker (m mod n)
```

Here, `n` is the current number of CR-layer workers. If the auto-tuner changes `n` from 4 to 2, the clients do not need to change. The server updates a global variable and lets workers switch roles at safe points.

The implementation also uses Multi-Packet Receive Queue (MP-RQ) to reduce receive-posting overhead. In an ordinary receive queue, a receive work request often corresponds to one small request. With MP-RQ, a larger receive buffer can hold multiple packets or requests, so the management thread posts receives less frequently.

### Resizable Cache

The CR layer caches hot items. The paper avoids conventional LRU because its bookkeeping overhead is too high for a high-speed KVS. Instead, μTPS uses a hot-set-based approach: a background thread samples recently accessed keys, uses count-min sketch and a min heap to identify hot keys, and switches the cache contents with an epoch-based mechanism.

The cache is resizable. A cache that is too small misses important hot items, while a cache that is too large consumes LLC capacity and can hurt miss-path processing. The auto-tuner therefore adjusts how many hot-set items are actually managed by the CR layer.

### Memory-Resident Layer and Batched Indexing

The MR layer manages the full index and full data items. It processes requests forwarded from the CR layer. Since this layer performs many random memory accesses, μTPS uses batched indexing to hide cache-miss latency.

Batched indexing is not merely calling `lookup` on a batch of keys. Its key idea is to interleave the pointer chasing of multiple index lookups:

```text
request A: prefetch next node, yield
request B: prefetch next node, yield
request C: prefetch next node, yield
return to A: the prefetched cache line may have arrived
```

μTPS represents each indexing operation as a C++ coroutine. An MR worker pops a batch of requests from the CR-MR queue, creates or resumes an indexing coroutine for each request, and switches to another coroutine after a prefetch. This uses computation from other requests to hide DRAM latency.

### CR-MR Queue

The CR-MR queue handles communication between the CR and MR layers. It is a multi-producer, multi-consumer structure. Each pair of CR thread and MR thread has a dedicated lock-free ring buffer. CR threads forward miss requests to MR threads in round-robin order to balance MR-layer load.

Each queue slot can hold multiple requests, which amortizes the fixed cost of inter-core communication, queue push/pop operations, and completion notification. Figure 12 shows that batching improves the throughput of μTPS-T and μTPS-H by 51.6% and 93.7%, respectively.

### Auto-Tuner

Because μTPS pins worker threads to cores, it cannot rely on the OS scheduler to rebalance resources across stages. The storage software itself includes an auto-tuner that adjusts:

- the number of worker threads assigned to the CR and MR layers;
- LLC-way allocation between the two layers;
- the size of the CR-layer hot cache.

The auto-tuner uses throughput feedback and hierarchical search. For a given cache size, it searches for a good thread allocation; then it probes different cache sizes. LLC allocation is tuned separately. In the dynamic-workload experiment, the auto-tuner finishes reconfiguration in about 0.9 seconds while request processing continues.

## Implementation and Evaluation

The paper implements two KVSs:

- **μTPS-H** uses libcuckoo as a hash index and mainly supports point queries.
- **μTPS-T** uses MassTree as a tree index and supports both point and range queries.

The testbed uses a 200 Gbps RDMA network. The server has Intel Xeon Gold 6330 CPUs, and most experiments use 28 cores on one NUMA node. Baselines include BaseKV, eRPC-KV, RaceHash, and Sherman. BaseKV uses the same optimizations as μTPS but keeps the run-to-completion thread model.

The main results are:

- μTPS achieves 1.03x to 5.46x speedup over the run-to-completion BaseKV.
- The gains are larger under skewed workloads, read-intensive workloads, tree indexes, and larger item sizes.
- μTPS-T often benefits more than μTPS-H because MassTree traversal involves more pointer chasing and cache misses than hash-table lookup.
- Under uniform workloads, hash indexes, small items, or very lightweight lookups, the gains are smaller; in some cases, eRPC-KV can approach or exceed μTPS.
- The extra latency introduced by μTPS is small. Inter-core communication is around 100 ns and is further amortized by batching.
- Figure 12 shows the importance of batching: as batch size increases from 1 to larger values, μTPS-T rises from about 30 Mops/s to about 45 Mops/s, and μTPS-H rises from about 37 Mops/s to about 72 Mops/s.

## Limitations

μTPS's benefits depend on workload and hardware. It is best suited for high-speed user-space KVSs with kernel-bypass networking, high request rates, substantial cache misses, or write contention. For uniform workloads, hash indexes, small values, and cheap lookups, run-to-completion designs or eRPC-style systems are already strong, and μTPS's extra inter-layer communication can offset part of its benefit.

The design also depends on RDMA, DDIO, CAT, core pinning, and related hardware/software features. Its complexity is higher than that of a conventional KVS because it includes reconfigurable RPC, a CR-MR queue, a resizable hot cache, and an auto-tuner. Although reconfiguration does not stop request processing, the roughly 0.9-second convergence time suggests that it is more suitable for workloads whose characteristics do not change too frequently.

## Questions and Clarifications During Reading

My first clarification was what a KVS thread means. In this paper, a thread is a server-side worker thread inside the KVS process, not a client. Clients issue `GET` and `PUT` requests. Server workers poll network queues, access the in-memory index and value storage, and send responses.

I also clarified that the KVS in this paper runs in user space rather than kernel space. The kernel and drivers are still involved in setup, such as creating RDMA queues, registering memory regions, and pinning memory. However, the request-processing fast path bypasses the traditional kernel socket stack.

For the difference between traditional TPS and μTPS, my understanding is that traditional TPS uses DRAM queues and buffers to hide slow I/O such as disks and older networks. μTPS faces modern RNICs and in-memory storage, where I/O is already fast and the bottlenecks are cache misses, contention, and inter-core communication. Therefore, μTPS reuses stage separation for cache isolation rather than for slow-I/O hiding.

For the sentence about the OS scheduler, my understanding is that traditional TPS relies on the OS scheduler to run and preempt ordinary thread-pool workers, which works well when threads block or sleep on slow I/O. μTPS workers are pinned to cores and busy-poll, so μTPS must decide in user space which cores belong to the CR layer and which cores belong to the MR layer.

For reconfigurable RPC, the key point is not that RPC semantics change. The key point is that the server receive path can change the number of workers without notifying clients. The shared receive buffer and slot-modulo assignment make the worker count a server-local variable.

For batched indexing, my understanding is that coroutines interleave multiple index traversals. Prefetch and yield hide DRAM latency from pointer chasing. This is different from ordinary batching because it targets the memory stalls inside index traversal, not only queue or function-call overhead.

## Takeaways

The value of this paper is that it shifts the bottleneck analysis of high-speed KVSs from network and I/O to cache behavior and thread architecture. In modern RDMA-based KVSs, run-to-completion reduces OS overhead, but it also mixes network buffers, hot items, full indexes, and data access inside the same worker. This lets cache-friendly and cache-unfriendly work interfere with each other.

μTPS follows a clear design logic:

- network buffers and hot items should be managed by the CR layer and kept cache-resident when possible;
- full indexes and data items should be managed by the MR layer, with batching, prefetching, and coroutines to hide memory latency;
- the CR-MR queue and batching control the cost introduced by stage separation;
- reconfigurable RPC lets the CR-layer worker count change without involving clients;
- the auto-tuner reallocates pinned cores in user space as workload characteristics change.

Overall, μTPS is a systems architecture paper. It does not only optimize a KVS data structure. It challenges the default assumption that one worker should process a request from beginning to end. Its broader lesson is that when devices become fast enough, the thread model itself determines cache locality, contention, and schedulability.
