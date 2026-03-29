for test in build/tests/svd_*; do
    ./"$test"
done

for test in build/tests/quantization_*; do
    ./"$test"
done

for test in build/tests/topk_*; do
    ./"$test"
done