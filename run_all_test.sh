for test in build/tests/svd_*; do
    ./"$test"
done

for test in build/tests/quantization_*; do
    ./"$test"
done

for test in build/tests/quantization_cuda_*; do
    ./"$test"
done

for test in build/tests/topk_cuda_*; do
    ./"$test"
done

for test in build/tests/bitsqz_llm_*; do
    ./"$test"
done
