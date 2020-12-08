find_package(Bullet REQUIRED Robotics)

add_library(
  pybulletphysics STATIC
  CBulletPhysicsManager.cpp CBulletPhysicsManager.h
)

target_link_libraries(
  pybulletphysics
  PUBLIC assets Bullet::Robotics
)

## Enable physics profiling
#add_compile_definitions(BT_ENABLE_PROFILE=0)
#add_definitions(-DBT_ENABLE_PROFILE)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
