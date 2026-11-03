# Minimal source subset of the local Bullet 3.27 snapshot.
set(BULLET_PHYSICS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" BULLET_VERSION)
set(INSTALL_LIBS OFF)
if(MSVC)
    # Bullet's own CMake build compiles with /fp:fast on MSVC (USE_MSVC_FAST_FLOATINGPOINT=ON).
    # Keep it: physics-driven bones, and therefore the extracted body targets, depend on it, and
    # the targets used for the paper were produced with that setting.
    add_compile_options(/fp:fast)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS)
endif()
add_subdirectory(src/LinearMath)
add_subdirectory(src/BulletCollision)
add_subdirectory(src/BulletDynamics)
