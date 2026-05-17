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
