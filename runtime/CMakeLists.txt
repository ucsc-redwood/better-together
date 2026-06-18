# runtime/ -- the app-agnostic pipeline runtime (worker/SPSC ring, UMA buffers).
# Header-only today; its only target is the schedule-contract test.
if(BT_BUILD_TESTS)
  # Pipeline schedule contract: chunks must contiguously cover [1, n_stages] -- the
  # correctness gate the pipeline path was missing (catches the dropped-stage class).
  bt_add_omp_test(test-schedule-omp tests/test_schedule.cpp)
  # Cross-tool round-trip contract: the C++ consumer reads the committed schedule
  # fixture the Python producer/schema side also checks (Lever B).
  target_compile_definitions(test-schedule-omp PRIVATE
    BT_SCHEDULE_FIXTURE="${CMAKE_SOURCE_DIR}/tests/fixtures/schedule.contract.json")
  set_tests_properties(test-schedule-omp PROPERTIES LABELS "omp;unit")
endif()
