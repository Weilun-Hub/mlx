echo "+-------------------------------------------+"
echo "|Building mlx                               |"
echo "+-------------------------------------------+"
python setup.py build_ext --inplace

# cd python/tests
# echo "+-------------------------------------------+"
# echo "|Running tests scaled_dot_product_attention |"
# echo "+-------------------------------------------+"
# python test_fast_sdpa_simple.py

# echo "+-------------------------------------------+"
# echo "|Running tests infllmv2_attention_stage1    |"
# echo "+-------------------------------------------+"
# python test_infllmv2_attn_stage1.py

# echo "+-------------------------------------------+"
# echo "|Running tests maxpooling                   |"
# echo "+-------------------------------------------+"
# python test_maxpooling.py

# echo "+-------------------------------------------+"
# echo "|Running tests topk                         |"
# echo "+-------------------------------------------+"
# python test_topk.py

# echo "+-------------------------------------------+"
# echo "|Running tests topk_to_uint64               |"
# echo "+-------------------------------------------+"
# python test_topk_to_uint64.py

# echo "+-------------------------------------------+"
# echo "|Running tests infllmv2_attn_stage2          |"
# echo "+-------------------------------------------+"
# python test_infllmv2_attn_stage2.py

echo "+-------------------------------------------+"
echo "|Done!                                      |"
echo "+-------------------------------------------+"