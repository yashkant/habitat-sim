// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"
//#include "BulletCollision/Gimpact/btGImpactShape.h"

#include "CBulletPhysicsManager.h"
//#include "esp/assets/ResourceManager.h"

namespace esp {
namespace physics {

CBulletPhysicsManager::~CBulletPhysicsManager() {
  LOG(INFO) << "Deconstructing CBulletPhysicsManager";

  existingObjects_.clear();
  staticStageObject_.reset(nullptr);
}

}  // namespace physics
}  // namespace esp
