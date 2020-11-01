// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "LightSetup.h"

namespace esp {
namespace gfx {

bool operator==(const LightInfo& a, const LightInfo& b) {
  return a.vector == b.vector && a.color == b.color && a.model == b.model;
}

bool operator!=(const LightInfo& a, const LightInfo& b) {
  return !(a == b);
}

Magnum::Vector4 getLightPositionRelativeToCamera(
    const LightInfo& light,
    const Magnum::Matrix4& transformationMatrix,
    const Magnum::Matrix4& cameraMatrix) {
  CORRADE_ASSERT(light.vector.w() == 1 || light.vector.w() == 0,
                 "Light vector"
                     << light.vector
                     << "is expected to have w == 0 for a directional light or "
                        "w == 1 for a point light",
                 {});

  switch (light.model) {
    case LightPositionModel::OBJECT:
      return transformationMatrix * light.vector;
    case LightPositionModel::GLOBAL:
      return cameraMatrix * light.vector;
    case LightPositionModel::CAMERA:
      return light.vector;
  }

  CORRADE_INTERNAL_ASSERT_UNREACHABLE();
}

LightSetup getLightsAtBoxCorners(const Magnum::Range3D& box,
                                 const Magnum::Color3& lightColor) {
  // NOLINTNEXTLINE(google-build-using-namespace)
  using namespace Magnum::Math::Literals;

  constexpr float w = 1;
  return LightSetup{{{box.frontTopLeft(), w}, lightColor},
                    {{box.frontTopRight(), w}, lightColor},
                    {{box.frontBottomLeft(), w}, lightColor},
                    {{box.frontBottomRight(), w}, lightColor},
                    {{box.backTopLeft(), w}, lightColor},
                    {{box.backTopRight(), w}, lightColor},
                    {{box.backBottomLeft(), w}, lightColor},
                    {{box.backBottomRight(), w}, lightColor}};
}

LightSetup getDefaultLights() {
  return getDefaultThreeLights();
  // XXX
  return LightSetup{{{1.0, 1.0, 0.0, 0.0}, {0.75, 0.75, 0.75}},
                    {{-0.5, 0.0, 1.0, 0.0}, {0.4, 0.4, 0.4}}};
}

/**
 * @brief Get a @ref LightSetup with three point lighting (key, fill rim).
 * Three lights are directional lights in camera space.
 * Key lights:
 *   direction: (1, -1, -1) (from position (-1, 1, 1) to origin)
 * Fill lights:
 *   direction: (-1, 1, -1) (from position (1, -1, 1) to origin)
 *   It is the weakest among the three.
 * Rim lights:
 *   direction: (0, 0, 1).
 *   It is the strongest among the three.
 */

LightSetup getDefaultThreeLights() {
  return LightSetup{
      /*
      {{1.0, -1.0, -1.0, 0.0},
       {2, 2, 2},
       LightPositionModel::CAMERA},  // Key light
      {{-1.0, 1.0, -1.0, 0.0},
       {0.4, 0.4, 0.4},
       LightPositionModel::CAMERA},  // Fill light
      {{0.0, 0.0, 1.0, 0.0},
       {0.5, 0.5, 0.5},
       LightPositionModel::CAMERA},  // Rim light
       */

      {{10.0f, 10.0f, 10.0f, 0.0},
       {1.25, 1.25, 1.25},
       // {1.35, 1.35, 1.35},    // antique camera
       LightPositionModel::CAMERA},  // Key light
      {{-5.0f, -5.0f, 10.0f, 0.0},
       // {0.8, 0.8, 0.8},         // antique camera
       {0.8, 0.8, 0.8},              // flying helmet
       LightPositionModel::CAMERA},  // Fill light
      {{0.0f, 10.0f, -10.0f, 0.0},
       // {0.5, 0.5, 0.5},        // antique camera
       {0.1, 0.1, 0.1},              // flying helmet
       LightPositionModel::CAMERA},  // Rim light
  };
}

Magnum::Color3 getAmbientLightColor(const LightSetup& lightSetup) {
  if (lightSetup.size() == 0) {
    // We assume an empty light setup means the user wants "flat" shading,
    // meaning object ambient color should be copied directly to pixels as-is.
    // We can achieve this in the Phong shader using an ambient light color of
    // (1,1,1) and no additional light sources.
    return Magnum::Color3(1.0, 1.0, 1.0);
  } else {
    // todo: add up ambient terms from all lights in lightSetup
    // temp: hard-coded ambient light tuned for ReplicaCAD
    float ambientIntensity = 0.4;
    return Magnum::Color3(ambientIntensity, ambientIntensity, ambientIntensity);
  }
}

}  // namespace gfx
}  // namespace esp
