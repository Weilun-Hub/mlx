echo "+-------------------------------------------+"
echo "|Building mlx                               |"
echo "+-------------------------------------------+"
python setup.py build_ext --inplace

# echo "+-------------------------------------------+"
# echo "|Running tests scaled_dot_product_attention |"
# echo "+-------------------------------------------+"
# python python/tests/test_fast_sdpa_simple.py

# echo "+-------------------------------------------+"
# echo "|Running tests infllmv2_attention_stage1    |"
# echo "+-------------------------------------------+"
# python python/tests/test_infllmv2_attn_stage1.py

echo "+-------------------------------------------+"
echo "|Running tests maxpooling                   |"
echo "+-------------------------------------------+"
python python/tests/test_maxpooling.py

echo "+-------------------------------------------+"
echo "|Done!                                      |"
echo "+-------------------------------------------+"