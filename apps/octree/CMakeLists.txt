# octree -- structural variant. test_sort.cpp is the one suite with real correctness
# checks (is_sorted + permutation oracle); it links bt::core (the live octree pipeline
# is the `tree` app -- the standalone octree app was removed, see git history).
if(BT_BUILD_TESTS)
    bt_add_omp_test(test-sort-omp omp/test_sort.cpp)
    set_tests_properties(test-sort-omp PROPERTIES LABELS "omp;differential")
endif()
