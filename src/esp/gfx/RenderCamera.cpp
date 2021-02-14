// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "RenderCamera.h"

#include <Magnum/EigenIntegration/Integration.h>
#include <Magnum/Math/Frustum.h>
#include <Magnum/Math/Intersection.h>
#include <Magnum/Math/Range.h>
#include <Magnum/SceneGraph/Drawable.h>
#include "esp/gfx/Drawable.h"
#include "esp/gfx/DrawableGroup.h"
#include "esp/scene/SceneGraph.h"

#ifdef ESP_BUILD_WITH_BULLET
#include "esp/physics/bullet/BulletRigidObject.h"

#include "esp/physics/bullet/BulletArticulatedObject.h"
#include "esp/physics/bullet/BulletURDFImporter.h"
#endif

namespace Mn = Magnum;
namespace Cr = Corrade;

namespace esp {
namespace gfx {

/**
 * @brief do frustum culling with temporal coherence
 * @param range, the axis-aligned bounding box
 * @param frustum, the frustum
 * @param frustumPlaneIndex, the frustum plane in last frame that culled the
 * aabb (default: 0)
 * @return NullOpt if aabb intersects the frustum, otherwise the fustum plane
 * that culls the aabb
 */
Cr::Containers::Optional<int> rangeFrustum(const Mn::Range3D& range,
                                           const Mn::Frustum& frustum,
                                           int frustumPlaneIndex = 0) {
  const Mn::Vector3 center = range.min() + range.max();
  const Mn::Vector3 extent = range.max() - range.min();

  for (int iPlane = 0; iPlane < 6; ++iPlane) {
    int index = (iPlane + frustumPlaneIndex) % 6;
    const Mn::Vector4& plane = frustum[index];

    const Mn::Vector3 absPlaneNormal = Mn::Math::abs(plane.xyz());

    const float d = Mn::Math::dot(center, plane.xyz());
    const float r = Mn::Math::dot(extent, absPlaneNormal);
    if (d + r < -2.0 * plane.w())
      return Cr::Containers::Optional<int>{index};
  }

  return Cr::Containers::NullOpt;
}

Cr::Containers::Optional<int> sphereFrustum(const Mn::Vector3& sphereCenter,
                                            const float sphereRadius,
                                            const Mn::Frustum& frustum,
                                            int frustumPlaneIndex = 0) {
  const float radiusSq = sphereRadius * sphereRadius;
  for (int iPlane = 0; iPlane < 6; ++iPlane) {
    int index = (iPlane + frustumPlaneIndex) % 6;

    const Mn::Vector4& plane = frustum[index];
    if (Mn::Math::Distance::pointPlaneScaled(sphereCenter, plane) < -radiusSq)
      return Cr::Containers::Optional<int>{index};
  }

  return Cr::Containers::NullOpt;
}

RenderCamera::RenderCamera(scene::SceneNode& node) : MagnumCamera{node} {
  node.setType(scene::SceneNodeType::CAMERA);
  setAspectRatioPolicy(Mn::SceneGraph::AspectRatioPolicy::NotPreserved);
}

RenderCamera::RenderCamera(scene::SceneNode& node,
                           const Mn::Vector3& eye,
                           const Mn::Vector3& target,
                           const Mn::Vector3& up)

    : RenderCamera(node) {
  // once it is attached, set the transformation
  resetViewingParameters(eye, target, up);
}

RenderCamera::RenderCamera(scene::SceneNode& node,
                           const vec3f& eye,
                           const vec3f& target,
                           const vec3f& up)
    : RenderCamera(node,
                   Mn::Vector3{eye},
                   Mn::Vector3{target},
                   Mn::Vector3{up}) {}

RenderCamera& RenderCamera::resetViewingParameters(const Mn::Vector3& eye,
                                                   const Mn::Vector3& target,
                                                   const Mn::Vector3& up) {
  this->node().setTransformation(Mn::Matrix4::lookAt(eye, target, up));
  return *this;
}

bool RenderCamera::isInSceneGraph(const scene::SceneGraph& sceneGraph) {
  return (this->node().scene() == sceneGraph.getRootNode().parent());
}

RenderCamera& RenderCamera::setProjectionMatrix(int width,
                                                int height,
                                                float znear,
                                                float zfar,
                                                Mn::Deg hfov) {
  const float aspectRatio = static_cast<float>(width) / height;
  auto projMat =
      Mn::Matrix4::perspectiveProjection(hfov, aspectRatio, znear, zfar);
  return setProjectionMatrix(width, height, projMat);
}

RenderCamera& RenderCamera::setOrthoProjectionMatrix(int width,
                                                     int height,
                                                     float znear,
                                                     float zfar,
                                                     float scale) {
  auto size = Mn::Vector2{width / (1.0f * height), 1.0f};
  size /= scale;
  auto orthoMat = Mn::Matrix4::orthographicProjection(size, znear, zfar);

  return setProjectionMatrix(width, height, orthoMat);
}

size_t RenderCamera::cull(
    std::vector<std::pair<std::reference_wrapper<Mn::SceneGraph::Drawable3D>,
                          Mn::Matrix4>>& drawableTransforms) {
  // camera frustum relative to world origin
  const Mn::Frustum frustum =
      Mn::Frustum::fromMatrix(projectionMatrix() * cameraMatrix());

  auto newEndIter = std::remove_if(
      drawableTransforms.begin(), drawableTransforms.end(),
      [&](const std::pair<std::reference_wrapper<Mn::SceneGraph::Drawable3D>,
                          Mn::Matrix4>& a) -> bool {
        // obtain the absolute aabb
        auto& node = static_cast<scene::SceneNode&>(a.first.get().object());
        Cr::Containers::Optional<int> culledPlane;
        Corrade::Containers::Optional<Mn::Range3D> aabb =
            node.getAbsoluteAABB();
#ifdef ESP_BUILD_WITH_BULLET
        if (!aabb) {
          // Guess that it is an articulated object

          const auto* parent = &node;
          // Need to go three levels up to find the node with the
          // ArticulatedLink feature
          for (int i = 0; i < 3 && parent; ++i) {
            parent = static_cast<const scene::SceneNode*>(parent->parent());
          }

          if (parent) {
            int i = 0;
            for (auto& abstractFeature : parent->features()) {
              auto link = dynamic_cast<const physics::BulletArticulatedLink*>(
                  &abstractFeature);
              if (link) {
                ++i;
                if (!aabb)
                  aabb = {link->getCollisionShapeAabb()};
              }
            }
            CORRADE_ASSERT(i == 0 || i == 1, "Didn't find 1 or 0 links", {});
          }

          // Otherwise try to see if it is a rigid
          if (!aabb && parent) {
            // Need to go up one more level to find a rigid node
            for (int i = 0; i < 1 && parent; ++i)
              parent = static_cast<const scene::SceneNode*>(parent->parent());

            if (parent) {
              int i = 0;
              for (auto& abstractFeature : parent->features()) {
                auto rigid = dynamic_cast<const physics::BulletRigidObject*>(
                    &abstractFeature);
                if (rigid) {
                  ++i;
                  if (!aabb)
                    aabb = {rigid->getRigidBodyAabb()};
                }
              }
              CORRADE_ASSERT(
                  i == 0 || i == 2,
                  "Didn't find two or zero rigids, that shouldn't happen...",
                  {});
            }
          }
        }
#endif
        if (aabb) {
          // if it has an absolute aabb, it is a static mesh
          culledPlane =
              rangeFrustum(*aabb, frustum, node.getFrustumPlaneIndex());
        } else {
          // Cull based on bounding sphere
          // Use diameter instead of radius as it isn't
          // clear where in the sphere node.absoluteTranslation() will end up
          // being. As long as that point is somewhere in the AABB (which it has
          // to be), this will be correct
          const float diameter = node.getCumulativeBB().size().length();
          culledPlane = sphereFrustum(node.absoluteTranslation(), diameter,
                                      frustum, node.getFrustumPlaneIndex());
        }
        if (culledPlane) {
          node.setFrustumPlaneIndex(*culledPlane);
        }
        // if it has value, it means the aabb is culled
        return (culledPlane != Cr::Containers::NullOpt);
      });

  return (newEndIter - drawableTransforms.begin());
}

size_t RenderCamera::removeNonObjects(
    std::vector<std::pair<std::reference_wrapper<Mn::SceneGraph::Drawable3D>,
                          Mn::Matrix4>>& drawableTransforms) {
  auto newEndIter = std::remove_if(
      drawableTransforms.begin(), drawableTransforms.end(),
      [&](const std::pair<std::reference_wrapper<Mn::SceneGraph::Drawable3D>,
                          Mn::Matrix4>& a) {
        auto& node = static_cast<scene::SceneNode&>(a.first.get().object());
        if (node.getType() == scene::SceneNodeType::OBJECT) {
          // don't remove OBJECT types
          return false;
        }
        return true;
      });
  return (newEndIter - drawableTransforms.begin());
}

uint32_t RenderCamera::draw(MagnumDrawableGroup& drawables, Flags flags) {
  previousNumVisibleDrawables_ = drawables.size();
  if (flags == Flags()) {  // empty set
    MagnumCamera::draw(drawables);
    return drawables.size();
  }

  if (flags & Flag::UseDrawableIdAsObjectId) {
    useDrawableIds_ = true;
  }

  std::vector<std::pair<std::reference_wrapper<Mn::SceneGraph::Drawable3D>,
                        Mn::Matrix4>>
      drawableTransforms = drawableTransformations(drawables);

  if (flags & Flag::ObjectsOnly) {
    // draw just the OBJECTS
    size_t numObjects = removeNonObjects(drawableTransforms);
    drawableTransforms.erase(drawableTransforms.begin() + numObjects,
                             drawableTransforms.end());
  }

  if (flags & Flag::FrustumCulling) {
    // draw just the visible part
    previousNumVisibleDrawables_ = cull(drawableTransforms);
    // erase all items that did not pass the frustum visibility test
    drawableTransforms.erase(
        drawableTransforms.begin() + previousNumVisibleDrawables_,
        drawableTransforms.end());
  }

  MagnumCamera::draw(drawableTransforms);

  // reset
  if (useDrawableIds_) {
    useDrawableIds_ = false;
  }
  return drawableTransforms.size();
}

esp::geo::Ray RenderCamera::unproject(const Mn::Vector2i& viewportPosition) {
  esp::geo::Ray ray;
  ray.origin = object().absoluteTranslation();

  const Magnum::Vector2i viewPos{viewportPosition.x(),
                                 viewport().y() - viewportPosition.y() - 1};

  const Magnum::Vector3 normalizedPos{
      2 * Magnum::Vector2{viewPos} / Magnum::Vector2{viewport()} -
          Magnum::Vector2{1.0f},
      1.0};

  ray.direction =
      ((object().absoluteTransformationMatrix() * projectionMatrix().inverted())
           .transformPoint(normalizedPos) -
       ray.origin)
          .normalized();
  return ray;
}

}  // namespace gfx
}  // namespace esp
