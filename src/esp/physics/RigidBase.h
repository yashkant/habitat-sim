// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_PHYSICS_RIGIDBASE_H_
#define ESP_PHYSICS_RIGIDBASE_H_

#include "esp/assets/Asset.h"
#include "esp/assets/BaseMesh.h"
#include "esp/assets/GenericInstanceMeshData.h"
#include "esp/assets/MeshData.h"
#include "esp/core/RigidState.h"
#include "esp/core/esp.h"
#include "esp/geo/VoxelWrapper.h"
#include "esp/metadata/attributes/AttributesBase.h"
#include "esp/physics/PhysicsObjectBase.h"

/** @file
 * @brief Class @ref esp::physics::Rigidbase
 */

namespace esp {
namespace assets {
class ResourceManager;
}
namespace metadata {
namespace attributes {
class AbstractObjectAttributes;
}  // namespace attributes
}  // namespace metadata

namespace physics {

class RigidBase : public esp::physics::PhysicsObjectBase {
 public:
  RigidBase(scene::SceneNode* rigidBodyNode,
            int objectId,
            const assets::ResourceManager& resMgr)
      : PhysicsObjectBase(rigidBodyNode, objectId, resMgr),
        visualNode_(&rigidBodyNode->createChild()) {}

  /**
   * @brief Virtual destructor for a @ref RigidBase.
   */
  ~RigidBase() override = default;

  /**
   * @brief Initializes the @ref RigidObject or @ref RigidStage that inherits
   * from this class.  This is overridden
   * @param initAttributes The template structure defining relevant phyiscal
   * parameters for this object
   * @return true if initialized successfully, false otherwise.
   */
  virtual bool initialize(
      metadata::attributes::AbstractObjectAttributes::ptr initAttributes) = 0;

  /**
   * @brief Finalize the creation of @ref RigidObject or @ref RigidStage that
   * inherits from this class.
   * @return whether successful finalization.
   */
  virtual bool finalizeObject() = 0;

 private:
  /**
   * @brief Finalize the initialization of this @ref RigidBase. This is
   * overridden by inheriting objects
   * @param resMgr Reference to resource manager, to access relevant components
   * pertaining to the object
   * @return true if initialized successfully, false otherwise.
   */
  virtual bool initialization_LibSpecific() = 0;
  /**
   * @brief any physics-lib-specific finalization code that needs to be run
   * after @ref RigidObject or @ref RigidStage is created.
   * @return whether successful finalization.
   */
  virtual bool finalizeObject_LibSpecific() = 0;

 public:
  bool getCollidable() const { return isCollidable_; }
  /**
   * @brief Set a rigid as collidable or not. Derived implementations handle the
   * specifics of modifying the collision properties.
   */
  virtual void setCollidable(CORRADE_UNUSED bool collidable){};

  /**
   * @brief Check whether object is being actively simulated, or sleeping.
   * Kinematic objects are always active, but derived dynamics implementations
   * may not be.  NOTE: no active objects without a physics engine...
   * (kinematics don't count)
   * @return true if active, false otherwise.
   */
  virtual bool isActive() const { return false; }

  /**
   * @brief Set an object as being actively simulated rather than sleeping.
   * Kinematic objects are always active, but derived dynamics implementations
   * may not be.
   */
  virtual void setActive() {}

  /**
   * @brief Apply a force to an object through a dervied dynamics
   * implementation. Does nothing for @ref MotionType::STATIC and @ref
   * MotionType::KINEMATIC objects.
   * @param force The desired force on the object in the global coordinate
   * system.
   * @param relPos The desired location of force application in the global
   * coordinate system relative to the object's center of mass.
   */
  virtual void applyForce(CORRADE_UNUSED const Magnum::Vector3& force,
                          CORRADE_UNUSED const Magnum::Vector3& relPos) {}

  /**
   * @brief Apply an impulse to an object through a dervied dynamics
   * implementation. Directly modifies the object's velocity without requiring
   * integration through simulation. Does nothing for @ref MotionType::STATIC
   * and @ref MotionType::KINEMATIC objects.
   * @param impulse The desired impulse on the object in the global coordinate
   * system.
   * @param relPos The desired location of impulse application in the global
   * coordinate system relative to the object's center of mass.
   */
  virtual void applyImpulse(CORRADE_UNUSED const Magnum::Vector3& impulse,
                            CORRADE_UNUSED const Magnum::Vector3& relPos) {}

  /**
   * @brief Apply an internal torque to an object through a dervied dynamics
   * implementation. Does nothing for @ref MotionType::STATIC and @ref
   * MotionType::KINEMATIC objects.
   * @param torque The desired torque on the object in the local coordinate
   * system.
   */
  virtual void applyTorque(CORRADE_UNUSED const Magnum::Vector3& torque) {}
  /**
   * @brief Apply an internal impulse torque to an object through a dervied
   * dynamics implementation. Does nothing for @ref MotionType::STATIC and @ref
   * MotionType::KINEMATIC objects.
   * @param impulse The desired impulse torque on the object in the local
   * coordinate system. Directly modifies the object's angular velocity without
   * requiring integration through simulation.
   */
  virtual void applyImpulseTorque(
      CORRADE_UNUSED const Magnum::Vector3& impulse) {}

  // ==== Transformations ===

  /** @brief Set the 4x4 transformation matrix of the object kinematically.
   * Calling this during simulation of a @ref MotionType::DYNAMIC object is not
   * recommended.
   * @param transformation The desired 4x4 transform of the object.
   */
  virtual void setTransformation(const Magnum::Matrix4& transformation) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().setTransformation(transformation);
      syncPose();
    }
  }

  virtual Magnum::Matrix4 getTransformation() const {
    return node().transformation();
  }

  /** @brief Set the 3D position of the object kinematically.
   * Calling this during simulation of a @ref MotionType::DYNAMIC object is not
   * recommended.
   * @param vector The desired 3D position of the object.
   */
  virtual void setTranslation(const Magnum::Vector3& vector) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().setTranslation(vector);
      syncPose();
    }
  }

  virtual Magnum::Vector3 getTranslation() const {
    return node().translation();
  }

  /** @brief Set the orientation of the object kinematically.
   * Calling this during simulation of a @ref MotionType::DYNAMIC object is not
   * recommended.
   * @param quaternion The desired orientation of the object.
   */
  virtual void setRotation(const Magnum::Quaternion& quaternion) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().setRotation(quaternion);
      syncPose();
    }
  }

  virtual Magnum::Quaternion getRotation() const { return node().rotation(); }

  /**
   * @brief Get the rotation and translation of the object.
   */
  virtual core::RigidState getRigidState() {
    return core::RigidState(node().rotation(), node().translation());
  };

  /**
   * @brief Set the rotation and translation of the object.
   */
  virtual void setRigidState(const core::RigidState& rigidState) {
    setTranslation(rigidState.translation);
    setRotation(rigidState.rotation);
  };

  /** @brief Reset the transformation of the object.
   * !!NOT IMPLEMENTED!!
   */
  virtual void resetTransformation() {
    if (objectMotionType_ != MotionType::STATIC) {
      node().resetTransformation();
      syncPose();
    }
  }

  /** @brief Modify the 3D position of the object kinematically by translation.
   * Calling this during simulation of a @ref MotionType::DYNAMIC object is not
   * recommended.
   * @param vector The desired 3D vector by which to translate the object.
   */
  virtual void translate(const Magnum::Vector3& vector) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().translate(vector);
      syncPose();
    }
  }

  /** @brief Modify the 3D position of the object kinematically by translation
   * with a vector defined in the object's local coordinate system. Calling this
   * during simulation of a @ref MotionType::DYNAMIC object is not recommended.
   * @param vector The desired 3D vector in the object's ocal coordiante system
   * by which to translate the object.
   */
  virtual void translateLocal(const Magnum::Vector3& vector) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().translateLocal(vector);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying an
   * axis-angle rotation to it. Calling this during simulation of a @ref
   * MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   * @param normalizedAxis The desired unit vector axis of rotation.
   */
  virtual void rotate(const Magnum::Rad angleInRad,
                      const Magnum::Vector3& normalizedAxis) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotate(angleInRad, normalizedAxis);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying an
   * axis-angle rotation to it in the local coordinate system. Calling this
   * during simulation of a @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   * @param normalizedAxis The desired unit vector axis of rotation in the local
   * coordinate system.
   */
  virtual void rotateLocal(const Magnum::Rad angleInRad,
                           const Magnum::Vector3& normalizedAxis) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateLocal(angleInRad, normalizedAxis);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying a
   * rotation to it about the global X axis. Calling this during simulation of a
   * @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   */
  virtual void rotateX(const Magnum::Rad angleInRad) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateX(angleInRad);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying a
   * rotation to it about the global Y axis. Calling this during simulation of a
   * @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   */
  virtual void rotateY(const Magnum::Rad angleInRad) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateY(angleInRad);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying a
   * rotation to it about the global Z axis. Calling this during simulation of a
   * @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   */
  virtual void rotateZ(const Magnum::Rad angleInRad) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateZ(angleInRad);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying a
   * rotation to it about the local X axis. Calling this during simulation of a
   * @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   */
  virtual void rotateXLocal(const Magnum::Rad angleInRad) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateXLocal(angleInRad);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying a
   * rotation to it about the local Y axis. Calling this during simulation of a
   * @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   */
  virtual void rotateYLocal(const Magnum::Rad angleInRad) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateYLocal(angleInRad);
      syncPose();
    }
  }

  /** @brief Modify the orientation of the object kinematically by applying a
   * rotation to it about the local Z axis. Calling this during simulation of a
   * @ref MotionType::DYNAMIC object is not recommended.
   * @param angleInRad The angle of rotation in radians.
   */
  virtual void rotateZLocal(const Magnum::Rad angleInRad) {
    if (objectMotionType_ != MotionType::STATIC) {
      node().rotateZLocal(angleInRad);
      syncPose();
    }
  }

  // ==== Getter/Setter functions ===

  //! For kinematic objects they are dummies, for dynamic objects
  //! implemented in physics-engine specific ways

  /** @brief Get the scalar angular damping coefficient of the object. Only used
   * for dervied dynamic implementations of @ref RigidObject.
   * @return The scalar angular damping coefficient of the object.
   */
  virtual double getAngularDamping() const { return 0.0; }

  /** @brief Set the scalar angular damping coefficient for the object. Only
   * used for dervied dynamic implementations of @ref RigidObject.
   * @param angDamping The new scalar angular damping coefficient for the
   * object.
   */
  virtual void setAngularDamping(CORRADE_UNUSED const double angDamping) {}

  /**
   * @brief Virtual angular velocity getter for an object.
   *
   * Returns zero for default @ref MotionType::KINEMATIC or @ref
   * MotionType::STATIC objects.
   * @return Angular velocity vector corresponding to world unit axis angles.
   */
  virtual Magnum::Vector3 getAngularVelocity() const {
    return Magnum::Vector3();
  };

  /** @brief Virtual angular velocity setter for an object.
   *
   * Does nothing for default @ref MotionType::KINEMATIC or @ref
   * MotionType::STATIC objects.
   * @param angVel Angular velocity vector corresponding to world unit axis
   * angles.
   */
  virtual void setAngularVelocity(
      CORRADE_UNUSED const Magnum::Vector3& angVel) {}

  /** @brief Get the center of mass (COM) of the object.
   * @return Object 3D center of mass in the global coordinate system.
   * @todo necessary for @ref MotionType::KINEMATIC?
   */
  virtual Magnum::Vector3 getCOM() const {
    const Magnum::Vector3 com = Magnum::Vector3();
    return com;
  }
  /** @brief Set the center of mass (COM) of the object.
   * @param COM Object 3D center of mass in the local coordinate system.
   * @todo necessary for @ref MotionType::KINEMATIC?
   */
  virtual void setCOM(CORRADE_UNUSED const Magnum::Vector3& COM) {}

  /** @brief Get the scalar friction coefficient of the object. Only used for
   * dervied dynamic implementations of @ref RigidObject.
   * @return The scalar friction coefficient of the object.
   */
  virtual double getFrictionCoefficient() const { return 0.0; }

  /** @brief Set the scalar friction coefficient of the object. Only used for
   * dervied dynamic implementations of @ref RigidObject.
   * @param frictionCoefficient The new scalar friction coefficient of the
   * object.
   */
  virtual void setFrictionCoefficient(
      CORRADE_UNUSED const double frictionCoefficient) {}

  /** @brief Get the 3x3 inertia matrix for an object.
   * @return The object's 3x3 inertia matrix.
   * @todo provide a setter for the full 3x3 inertia matrix. Not all
   * implementations will provide this option.
   */
  virtual Magnum::Matrix3 getInertiaMatrix() const {
    const Magnum::Matrix3 inertia = Magnum::Matrix3();
    return inertia;
  }

  /** @brief Get the diagonal of the inertia matrix for an object.
   * If an object is aligned with its principle axii of inertia, the 3x3 inertia
   * matrix can be reduced to a diagonal. See @ref
   * RigidObject::setInertiaVector.
   * @return The diagonal of the object's inertia matrix.
   */
  virtual Magnum::Vector3 getInertiaVector() const {
    const Magnum::Vector3 inertia = Magnum::Vector3();
    return inertia;
  }

  /** @brief Set the diagonal of the inertia matrix for the object.
   * If an object is aligned with its principle axii of inertia, the 3x3 inertia
   * matrix can be reduced to a diagonal.
   * @param inertia The new diagonal for the object's inertia matrix.
   */
  virtual void setInertiaVector(CORRADE_UNUSED const Magnum::Vector3& inertia) {
  }

  /** @brief Get a copy of the template used to initialize this object
   * or scene.
   * @return A copy of the initialization template used to create this object
   * instance or nullptr if no template exists.
   */
  template <class T>
  std::shared_ptr<T> getInitializationAttributes() const {
    if (!initializationAttributes_) {
      return nullptr;
    }
    return T::create(*(static_cast<T*>(initializationAttributes_.get())));
  }

  /**
   * @brief Set the light setup of this rigid.
   * @param lightSetupKey @ref gfx::LightSetup key
   */
  void setLightSetup(const std::string& lightSetupKey) {
    gfx::setLightSetupForSubTree(node(), lightSetupKey);
  }

  /** @brief Get the scalar linear damping coefficient of the object. Only used
   * for dervied dynamic implementations of @ref RigidObject.
   * @return The scalar linear damping coefficient of the object.
   */
  virtual double getLinearDamping() const { return 0.0; }

  /** @brief Set the scalar linear damping coefficient of the object. Only used
   * for dervied dynamic implementations of @ref RigidObject.
   * @param linDamping The new scalar linear damping coefficient of the object.
   */
  virtual void setLinearDamping(CORRADE_UNUSED const double linDamping) {}

  /**
   * @brief Virtual linear velocity getter for an object.
   *
   * Returns zero for default @ref MotionType::KINEMATIC or @ref
   * MotionType::STATIC objects.
   * @return Linear velocity of the object.
   */
  virtual Magnum::Vector3 getLinearVelocity() const {
    return Magnum::Vector3();
  };

  /**
   * @brief Virtual linear velocity setter for an object.
   *
   * Does nothing for default @ref MotionType::KINEMATIC or @ref
   * MotionType::STATIC objects.
   * @param linVel Linear velocity to set.
   */
  virtual void setLinearVelocity(CORRADE_UNUSED const Magnum::Vector3& linVel) {
  }

  /** @brief Get the mass of the object. Only used for dervied dynamic
   * implementations of @ref RigidObject.
   * @return The mass of the object.
   */
  virtual double getMass() const { return 0.0; }

  /** @brief Set the mass of the object. Only used for dervied dynamic
   * implementations of @ref RigidObject.
   * @param mass The new mass of the object.
   */
  virtual void setMass(CORRADE_UNUSED const double mass) {}

  /** @brief Get the scalar coefficient of restitution  of the object. Only used
   * for dervied dynamic implementations of @ref RigidObject.
   * @return The scalar coefficient of restitution  of the object.
   */
  virtual double getRestitutionCoefficient() const { return 0.0; }

  /** @brief Set the scalar coefficient of restitution of the object. Only used
   * for dervied dynamic implementations of @ref RigidObject.
   * @param restitutionCoefficient The new scalar coefficient of restitution of
   * the object.
   */
  virtual void setRestitutionCoefficient(
      CORRADE_UNUSED const double restitutionCoefficient) {}

  /** @brief Get the scale of the object set during initialization.
   * @return The scaling for the object relative to its initially loaded meshes.
   */
  virtual Magnum::Vector3 getScale() const {
    return initializationAttributes_->getScale();
  }

  /**
   * @brief Get the semantic ID for this object.
   */
  int getSemanticId() const { return visualNode_->getSemanticId(); }

  /**
   * @brief Set the @ref esp::scene::SceneNode::semanticId_ for all visual nodes
   * belonging to the object.
   * @param semanticId The desired semantic id for the object.
   */
  void setSemanticId(uint32_t semanticId) {
    for (auto node : visualNodes_) {
      node->setSemanticId(semanticId);
    }
  }

  /**
   * @brief Get pointers to this rigid's visual SceneNodes.
   * @return vector of pointers to the rigid's visual scene nodes.
   */
  std::vector<scene::SceneNode*> getVisualSceneNodes() const {
    return visualNodes_;
  }

  /** @brief Get the VoxelWrapper for the object.
   * @return The voxel wrapper for the object.
   */
  std::shared_ptr<esp::geo::VoxelWrapper> getVoxelization() const {
    return voxelWrapper;
  }

#ifdef ESP_BUILD_WITH_VHACD

  /** @brief Initializes a new VoxelWrapper with a specified resolution. Creates
   * a boundary voxelization (registered under the key "Boundary" in the
   * VoxelGrid) using VHACD.
   * @param resourceManager_ A reference to the current resource manager, used
   * for registering the newly created voxel grid within the resource manager's
   * VoxelGrid dictionary.
   * @param resolution Represents the approximate number of voxels in the new
   * voxelization.
   */
  void generateVoxelization(esp::assets::ResourceManager& resourceManager_,
                            int resolution = 1000000) {
    std::string renderAssetHandle =
        initializationAttributes_->getRenderAssetHandle();
    voxelWrapper =
        std::make_shared<esp::geo::VoxelWrapper>(esp::geo::VoxelWrapper(
            renderAssetHandle, &node(), resourceManager_, resolution));
  }
#endif

  /** @brief Store whatever object attributes you want here! */
  esp::core::Configuration::ptr attributes_{};

  //! The @ref SceneNode of a bounding box debug drawable. If nullptr, BB
  //! drawing is off. See @ref setObjectBBDraw().
  scene::SceneNode* BBNode_ = nullptr;

  //! The @ref SceneNode of the voxel drawable. If nullptr, Voxel
  //! drawing is off. See @ref setObjectVoxelizationDraw().
  scene::SceneNode* VoxelNode_ = nullptr;

  /**
   * @brief All Drawable components are children of this node.
   *
   * Note that the transformation of this node is a composition of rotation and
   * translation as scaling is applied to a child of this node.
   */
  scene::SceneNode* visualNode_ = nullptr;

  //! all nodes created when this object's render asset was added to the
  //! SceneGraph
  std::vector<esp::scene::SceneNode*> visualNodes_;

  //! ptr to the VoxelWrapper associated with this RigidBase
  std::shared_ptr<esp::geo::VoxelWrapper> voxelWrapper = nullptr;

 protected:
  /**
   * @brief Shift the object's local origin by translating all children of this
   * object's SceneNode.
   * @param shift The translation to apply to object's children.
   */
  virtual void shiftOrigin(const Magnum::Vector3& shift) {
    // shift visual components
    if (visualNode_)
      visualNode_->translate(shift);
    node().computeCumulativeBB();
  }

  /**
   * @brief Shift the object's local origin to be coincident with the center of
   * it's bounding box, @ref cumulativeBB_. See @ref shiftOrigin.
   */
  void shiftOriginToBBCenter() {
    shiftOrigin(-node().getCumulativeBB().center());
  }

  /** @brief Used to synchronize other simulator's notion of the object state
   * after it was changed kinematically. Called automatically on kinematic
   * updates.*/
  virtual void syncPose() { return; }

  /** @brief Flag sepcifying whether or not the object has an active collision
   * shape.
   */
  bool isCollidable_ = false;

  /**
   * @brief Saved attributes when the object was initialized.
   */
  metadata::attributes::AbstractObjectAttributes::ptr
      initializationAttributes_ = nullptr;

 public:
  ESP_SMART_POINTERS(RigidBase)
};  // class RigidBase

}  // namespace physics
}  // namespace esp
#endif  // ESP_PHYSICS_RIGIDBASE_H_
