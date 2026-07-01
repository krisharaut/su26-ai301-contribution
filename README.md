# su26-ai301-contribution
# Contribution [#1]: [Enhancement] Integrate SIMD with ADC distance computation

**Contribution Number:** [4]  
**Student:** Krisha Raut  
**Issue:** https://github.com/opensearch-project/k-NN/issues/3150  
**Status:** Phase IV Complete

---

## Why I Chose This Issue

I've been wanting to get into lower-level performance work for a while, and this issue felt like a legitimate way in. SIMD optimization is something I kept running into when reading about how databases and ML runtimes actually squeeze performance out of hardware, but I never had a real reason to dig into it. This gave me one. The fact that it involves both C++ and Java through JNI made it more interesting to me as it's not just writing fast code, it's understanding how that fast code gets called from a higher-level system.

The OpenSearch kNN plugin also just feels relevant right now. Vector search is everywhere, every RAG system, every semantic search product runs something like this under the hood. I'd rather contribute to infrastructure that people actually use than pick a safer, more isolated issue. I expect to come out of this with a much stronger mental model of how hardware-aware software gets built and shipped.

---

## Understanding the Issue

### Problem Description

Vectorizes the two ADC (Asymmetric Distance Computation) distance functions in
`KNNScoringUtil` — `l2SquaredADC` and `innerProductADC` — using the Java Vector
API (`jdk.incubator.vector`). Both methods previously ran a scalar loop over one
dimension at a time; they now process the query vector in
`FloatVector.SPECIES_PREFERRED`-wide chunks and reduce with `reduceLanes(ADD)`,
with a scalar tail loop for dimensions that don't fill a full lane. This closes
the TODOs the original author left in both methods.

### Why this change

The float/float and byte/byte distance paths already get SIMD acceleration for
free by delegating to Lucene's `VectorUtil`. The ADC methods can't do that — their
input is a full-precision float query vector against a bit-packed binary document
vector (one bit per dimension), which Lucene has no primitive for — so they were
left as a correct-but-slow scalar first pass with explicit TODOs pointing at
`FloatVector.SPECIES_PREFERRED` and `reduceLanes()`. This PR implements that.

### Issues Resolved

Closes #3150

### Testing

Added parity unit tests in `KNNScoringUtilTests` that assert the vectorized
results match the original scalar computation within a float tolerance, across
dimensions that are and aren't multiples of the SIMD lane width (to exercise the
tail loop), plus all-zero and all-one binary vectors as edge cases.


### Check List
- [ ] New functionality includes testing.
- [ ] All tests pass.
- [ ] New functionality has been documented / CHANGELOG updated.
- [ ] Commits are signed per the DCO using `--signoff`.


### Expected Behavior

l2SquaredADC and innerProductADC should compute their distances using SIMD via the Java Vector API — processing multiple dimensions per CPU instruction with FloatVector.SPECIES_PREFERRED and reducing the accumulated lanes with reduceLanes(ADD). This mirrors how the non-ADC distance paths already get their speed: the float/float and byte/byte versions of l2Squared and innerProduct delegate to Lucene's VectorUtil, which is SIMD-accelerated internally. The ADC methods should reach comparable throughput on modern hardware while returning the same numerical results.

### Current Behavior

Both methods run a plain scalar for loop over queryVector.length, handling one dimension per iteration. For each dimension i, the code computes the byte index (i / 8) and bit offset (7 - (i % 8)), extracts a single bit from the packed binary vector, and then either squares the difference (l2SquaredADC) or accumulates the product (innerProductADC). There's no batching and no vectorization — the TODO comment inside each method explicitly names this as inefficient and points to FloatVector.SPECIES_PREFERRED + reduceLanes() as the intended fix.

### Affected Components

src/main/java/org/opensearch/knn/plugin/script/KNNScoringUtil.java — holds l2SquaredADC, innerProductADC, and scoreWithADC (the dispatcher that routes the L2, inner product, and cosine space types to the two ADC methods).

build.gradle — needs the --add-modules=jdk.incubator.vector JVM argument wired into the build/test config so the incubator Vector API resolves at compile and runtime, as the TODO comments call out.

src/test/java/org/opensearch/knn/plugin/script/KNNScoringUtilTests.java — where the existing scoring-util tests live, so the new ADC correctness/parity tests belong there.

---

## Reproduction Process

### Environment Setup

This one took a bit more time to get running than a typical project — OpenSearch is a full distributed search engine, so there's more moving parts. I cloned my fork, set up the Java environment (the project requires JDK 21), and ran ./gradlew build -x test to get the initial build going. Hit a few issues with missing --add-modules jdk.incubator.vector flags not being set up yet, which is actually related to the issue itself — the TODO comments in the code mention this exact flag as a prerequisite for enabling the Vector API. Got the test suite running with ./gradlew test.

### Steps to Reproduce

1. Clone the repo and navigate to src/main/java/org/opensearch/knn/plugin/script/KNNScoringUtil.java
2. Look at l2SquaredADC (lines ~114–130) and innerProductADC (lines ~145–175) — both loop through each dimension one at a time with plain scalar arithmetic
3. The TODO comments in both methods explicitly flag this as inefficient and call out SIMD via FloatVector.SPECIES_PREFERRED as the intended fix
4. Expected: The ADC distance methods process multiple dimensions per CPU instruction using the Java Vector API, getting meaningful throughput improvement on modern hardware
5. Actual: Both methods use a scalar loop — no vectorization, no SIMD, processing one element at a time

The "bug" here is less a crash and more a performance gap that the original author left with explicit TODOs. Running a microbenchmark (or just reading the code) makes it clear there's no batching happening.

---

## Solution Approach

### Analysis

This isn't a bug in the usual sense — there's no crash or wrong answer. The root cause is simply that the ADC methods were written as a correct-but-slow first pass, with the optimization deliberately deferred (the TODO comments say as much). The scalar loop handles one dimension per iteration, so it can't take advantage of modern CPUs that process 4, 8, or 16 floats in a single SIMD instruction. The reason these methods couldn't just borrow the existing fast path is the input format: the query is a full-precision float[] while the document vector is bit-packed (one bit per dimension, MSB-first within each byte). Lucene's VectorUtil — which the float/float and byte/byte distance methods delegate to for their speed — has no primitive for that mixed float-vs-packed-bits case, so ADC fell back to a hand-written scalar loop. The speedup therefore hinges on cheaply turning a run of packed bits into a FloatVector of 0.0/1.0 values so it can be combined with a chunk of the query vector in one vector operation and then reduced.

### Proposed Solution

Rewrite both methods to iterate in chunks of FloatVector.SPECIES_PREFERRED.length() instead of one element at a time. For each chunk: load the matching slice of the query vector into a FloatVector, materialize the corresponding bits as a FloatVector of 0/1 values, and then — for l2SquaredADC, subtract, square, and accumulate into a running vector; for innerProductADC, multiply and accumulate — finishing each with reduceLanes(ADD). A scalar tail loop handles the leftover dimensions when the length isn't a multiple of the lane width, using the same math as the original so results stay identical. Alongside the code, add --add-modules=jdk.incubator.vector to build.gradle so the incubator API resolves. The dispatcher scoreWithADC stays untouched since it only forwards to these two methods.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** ADC (Asymmetric Distance Computation) is a technique used during vector search where one vector is stored in compressed binary format and the other is a full-precision float query vector. The distance computation has to unpack bits one at a time right now, which is slow. The Java Vector API (available since JDK 16 as an incubator module, stabilized in JDK 21) lets you do the same math on multiple floats simultaneously using SIMD instructions. The TODOs in l2SquaredADC and innerProductADC are asking for exactly this.

**Match:** The existing l2Squared and innerProduct methods for float-float and byte-byte cases already delegate to VectorUtil from Apache Lucene, which handles SIMD internally. The ADC methods can't use those because the mixed float/binary input format isn't supported there — so this needs a custom implementation using jdk.incubator.vector (or the stable java.lang.foreign / Vector API in JDK 21+). Looking at how other performance-sensitive parts of the codebase use Lucene's VectorUtil gives a good sense of the pattern.

**Plan:** 
1. Add the --add-modules jdk.incubator.vector JVM flag and any needed build.gradle config as the TODO comments suggest (this may already exist elsewhere in the project — worth checking)
2. Rewrite l2SquaredADC to use FloatVector.SPECIES_PREFERRED, processing the query vector in chunks matching the SIMD lane width, and accumulating the squared differences with reduceLanes(ADD)
3. Do the same for innerProductADC — load chunks of the query vector, multiply by the corresponding unpacked bit values, and reduce
4. Keep the scalar fallback path for dimensions that don't divide evenly into the lane width (the tail loop)
5. Write unit tests comparing the SIMD and scalar outputs for correctness, and ideally a JMH microbenchmark to show the throughput improvement

**Implement:** https://github.com/krisharaut/k-NN

**Review:** Will read CONTRIBUTING.md and the project's PR template before opening. This is a performance change so I'll make sure the PR includes before/after benchmark numbers — that seems like the kind of thing maintainers on a project like this will ask for anyway.

**Evaluate:** Unit tests checking that l2SquaredADC and innerProductADC return the same values as the old scalar implementation across various vector sizes (including non-power-of-two lengths to test the tail loop). Then a JMH benchmark on a reasonably large vector dimension (e.g. 768 or 1536, typical embedding sizes) to show wall-clock improvement.

---

## Testing Strategy
What I added in KNNScoringUtilTests.java:

1. L2 parity test: keeps a copy of the original scalar loop as a reference oracle and asserts the new SIMD l2SquaredADC matches it within 1e-4f, across dimensions both divisible and not divisible by the lane width.
2. Inner product parity test: same approach for innerProductADC, including all-zeros and all-ones binary vectors.
3. Boundary test: dimensions 8, 768, 1536, and an awkward value like 1000 to exercise chunk + tail handling.

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week 3 Progress

What I built:

1. Added --add-modules=jdk.incubator.vector to the test/build JVM args in build.gradle so the incubator Vector API resolves. 
2. Rewrote l2SquaredADC in KNNScoringUtil.java to process the query vector in FloatVector.SPECIES_PREFERRED.length()-wide chunks, accumulating squared differences and finishing with reduceLanes(ADD), with a scalar tail loop for the remainder. 
3. Same rewrite for innerProductADC.

Files modified:

1. src/main/java/org/opensearch/knn/plugin/script/KNNScoringUtil.java
2. build.gradle
3. src/test/java/org/opensearch/knn/plugin/script/KNNScoringUtilTests.java
4. CHANGELOG.md

**Challenges Faced:**

1. Unpacking bit-packed input into a float vector. The binary vector stores one bit per dimension, MSB-first (the original 7 - (i % 8) logic). Turning a run of lane-width bits into a FloatVector of 0.0/1.0 is the core difficulty, since there's no direct primitive for it. 
2. Tail loop off-by-one. When the dimension isn't a multiple of the lane width, the leftover elements need the original scalar math. 
3. Spotless / DCO failures on push. OpenSearch CI rejects unformatted code and unsigned commits. 

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:**
1. src/main/java/org/opensearch/knn/plugin/script/KNNScoringUtil.java
2. build.gradle
3. src/test/java/org/opensearch/knn/plugin/script/KNNScoringUtilTests.java
4. CHANGELOG.md

- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
