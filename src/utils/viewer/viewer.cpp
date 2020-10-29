// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <stdlib.h>
#include <ctime>

#include <Magnum/configure.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#ifdef MAGNUM_TARGET_WEBGL
#include <Magnum/Platform/EmscriptenApplication.h>
#else
#include <Magnum/Platform/GlfwApplication.h>
#endif
#include <Magnum/PixelFormat.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Timeline.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/Image.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Shaders/Shaders.h>

#include "esp/gfx/RenderCamera.h"
#include "esp/gfx/Renderer.h"
#include "esp/nav/PathFinder.h"
#include "esp/scene/ObjectControls.h"
#include "esp/scene/SceneNode.h"

#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Assert.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>
#include <Magnum/DebugTools/Screenshot.h>
#include <Magnum/EigenIntegration/GeometryIntegration.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <sophus/so3.hpp>
#include "esp/core/Utility.h"
#include "esp/core/esp.h"
#include "esp/gfx/Drawable.h"
#include "esp/io/io.h"

#include "esp/scene/SceneConfiguration.h"
#include "esp/sim/Simulator.h"

#include "ObjectPickingHelper.h"
#include "esp/physics/configure.h"

constexpr float moveSensitivity = 0.1f;
constexpr float lookSensitivity = 11.25f;
constexpr float rgbSensorHeight = 1.5f;

// for ease of access
namespace Cr = Corrade;
namespace Mn = Magnum;

namespace {

//! return current time as string in format
//! "year_month_day_hour-minutes-seconds"
std::string getCurrentTimeString() {
  time_t now = time(0);
  tm* ltm = localtime(&now);
  return std::to_string(1900 + ltm->tm_year) + "_" +
         std::to_string(1 + ltm->tm_mon) + "_" + std::to_string(ltm->tm_mday) +
         "_" + std::to_string(ltm->tm_hour) + "-" +
         std::to_string(ltm->tm_min) + "-" + std::to_string(ltm->tm_sec);
}

using namespace Mn::Math::Literals;

enum MouseInteractionMode {
  LOOK,
  ADD,
  REMOVE,
  GRAB,
  THROW,

  NUM_MODES
};

std::string getEnumName(MouseInteractionMode mode) {
  switch (mode) {
    case (GRAB):
      return "GRAB";
      break;
    case (ADD):
      return "ADD";
      break;
    case (REMOVE):
      return "REMOVE";
      break;
    case (THROW):
      return "THROW";
      break;
    case (LOOK):
      return "LOOK";
      break;
    default:
      return "NONE";
  }
}

struct MouseGrabber {
  esp::core::RigidState targetFrame;
  float gripDepth;
  esp::sim::Simulator* simulator_;

  int mode = 0;  // left (0) or right (1) click.

  MouseGrabber(const Magnum::Vector3& clickPos,
               float _gripDepth,
               esp::sim::Simulator* sim) {
    simulator_ = sim;
    targetFrame.translation = clickPos;
    gripDepth = _gripDepth;
  }

  virtual ~MouseGrabber() {}

  virtual void updateTarget(const esp::core::RigidState& newTarget) {
    targetFrame = newTarget;
  }
};

struct MouseObjectKinematicGrabber : public MouseGrabber {
  int objectId;
  Magnum::Vector3 clickOffset;
  esp::physics::MotionType originalMotionType;

  //! Used to set object velocity to match state change. Clear after each
  //! simulation step.
  esp::core::RigidState previousState;
  MouseObjectKinematicGrabber(const Magnum::Vector3& clickPos,
                              float _gripDepth,
                              int _objectId,
                              esp::sim::Simulator* sim)
      : MouseGrabber(clickPos, _gripDepth, sim) {
    objectId = _objectId;
    Magnum::Vector3 origin = simulator_->getTranslation(objectId);
    clickOffset = origin - clickPos;
    originalMotionType = simulator_->getObjectMotionType(objectId);
    targetFrame.rotation = simulator_->getRotation(objectId);
    previousState = targetFrame;
    simulator_->setObjectMotionType(esp::physics::MotionType::KINEMATIC,
                                    objectId);
  }

  virtual ~MouseObjectKinematicGrabber() override {
    Corrade::Utility::Debug()
        << "~MouseObjectKinematicGrabber final origin pos: "
        << simulator_->getTranslation(objectId);
    simulator_->setObjectMotionType(originalMotionType, objectId);
  }

  virtual void updateTarget(const esp::core::RigidState& newTarget) override {
    Magnum::Vector3 objectOrigin = simulator_->getTranslation(objectId);
    simulator_->setTranslation(clickOffset + newTarget.translation, objectId);
    simulator_->setRotation(newTarget.rotation, objectId);
  }

  //! update
  void updateVelAndCache(float dt) {
    Mn::Debug{} << "vel before = " << simulator_->getLinearVelocity(objectId);
    simulator_->setLinearVelocity(
        (targetFrame.translation - previousState.translation) / dt, objectId);
    Mn::Debug{} << "vel after = " << simulator_->getLinearVelocity(objectId);
    previousState = targetFrame;
  }
};

class Viewer : public Mn::Platform::Application {
 public:
  explicit Viewer(const Arguments& arguments);

 private:
  void drawEvent() override;
  void viewportEvent(ViewportEvent& event) override;
  void mousePressEvent(MouseEvent& event) override;
  void mouseReleaseEvent(MouseEvent& event) override;
  void mouseMoveEvent(MouseMoveEvent& event) override;
  void mouseScrollEvent(MouseScrollEvent& event) override;

  MouseInteractionMode mouseInteractionMode = LOOK;

  std::unique_ptr<MouseGrabber> mouseGrabber_ = nullptr;

  void keyPressEvent(KeyEvent& event) override;

  /**
   * @brief Instance an object from an ObjectAttributes.
   * @param configHandle The handle referencing the object's template in the
   * ObjectAttributesManager.
   * @return The newly allocated object's ID for referencing it through the
   * Simulator API.
   */
  int addObject(const std::string& configHandle);

  /**
   * @brief Instance an object from an ObjectAttributes.
   * @param objID The unique ID referencing the object's template in the
   * ObjectAttributesManager.
   * @return The newly allocated object's ID for referencing it through the
   * Simulator API.
   */
  int addObject(int objID);

  /**
   * @brief Instance a random object based on an imported asset if one exists.
   * @return The newly allocated object's ID for referencing it through the
   * Simulator API.
   */
  int addTemplateObject();

  /**
   * @brief Instance a random object based on a primitive shape.
   * @return The newly allocated object's ID for referencing it through the
   * Simulator API.
   */
  int addPrimitiveObject();

  /**
   * @brief Throw a sphere from the camera origin in a direction.
   * @return The newly allocated object's ID for referencing it through the
   * Simulator API.
   */
  int throwSphere(Mn::Vector3 direction);

  void pokeLastObject();
  void pushLastObject();
  void torqueLastObject();
  void removeLastObject();
  void wiggleLastObject();
  void invertGravity();
  Mn::Vector3 randomDirection();

  //! string rep of time when viewer application was started
  std::string viewerStartTimeString = getCurrentTimeString();
  void screenshot();

  std::string helpText = R"(
==================================================
Welcome to the Habitat-sim C++ Viewer application!
==================================================
Mouse Functions:
----------------
  LEFT:
    Click and drag to rotate the agent and look up/down.
  RIGHT:
    (With 'enable-physics') Click a surface to instance a random primitive object at that location.
  SHIFT-RIGHT:
    Click a mesh to highlight it.

Key Commands:
-------------
  esc: Exit the application.
  'H': Display this help message.

  Agent Controls:
  'wasd': Move the agent's body forward/backward, left/right.
  'zx': Move the agent's body up/down.
  arrow keys: Turn the agent's body left/right and camera look up/down.
  '9': Randomly place agent on NavMesh (if loaded).
  'q': Query the agent's state and print to terminal.

  Utilities:
  'e' enable/disable frustum culling.
  'c' show/hide FPS overlay.
  'n' show/hide NavMesh wireframe.
  'i' Save a screenshot to "./screenshots/year_month_day_hour-minute-second/#.png"

  Object Interactions:
  SPACE: Toggle physics simulation on/off
  '.': Take a single simulation step if not simulating continuously.
  '8': Instance a random primitive object in front of the agent.
  'o': Instance a random file-based object in front of the agent.
  'u': Remove most recently instanced object.
  'b': Toggle display of object bounding boxes.
  'k': Kinematically wiggle the most recently added object.
  'p': (physics) Poke the most recently added object.
  'f': (physics) Push the most recently added object.
  't': (physics) Torque the most recently added object.
  'v': (physics) Invert gravity.
==================================================
  )";

  //! Print viewer help text to terminal output.
  void printHelpText() { Mn::Debug{} << helpText; };

  // single inline for logging agent state msgs, so can be easily modified
  inline void logAgentStateMsg(bool showPos, bool showOrient) {
    std::stringstream strDat("");
    if (showPos) {
      strDat << "Agent position "
             << Eigen::Map<esp::vec3f>(agentBodyNode_->translation().data())
             << " ";
    }
    if (showOrient) {
      strDat << "Agent orientation "
             << esp::quatf(agentBodyNode_->rotation()).coeffs().transpose();
    }

    auto str = strDat.str();
    if (str.size() > 0) {
      LOG(INFO) << str;
    }
  }

  // The simulator object backend for this viewer instance
  std::unique_ptr<esp::sim::Simulator> simulator_;

  // cached configuration used to reconfigure Simulator for easy recycling
  esp::sim::SimulatorConfiguration simConfig_;

  // Toggle physics simulation on/off
  bool simulating_ = true;

  // Toggle a single simulation step at the next opportunity if not simulating
  // continuously.
  bool simulateSingleStep_ = false;

  // The managers belonging to the simulator
  std::shared_ptr<esp::metadata::managers::ObjectAttributesManager>
      objectAttrManager_ = nullptr;
  std::shared_ptr<esp::metadata::managers::AssetAttributesManager>
      assetAttrManager_ = nullptr;
  std::shared_ptr<esp::metadata::managers::StageAttributesManager>
      stageAttrManager_ = nullptr;
  std::shared_ptr<esp::metadata::managers::PhysicsAttributesManager>
      physAttrManager_ = nullptr;
  std::shared_ptr<esp::metadata::managers::SceneAttributesManager>
      sceneAttrManager_ = nullptr;

  //! load a particular scene instance by handle
  void loadSceneInstance(std::string sceneInstanceHandle);

  //! used to dynamically swap instances
  int activeSceneInstanceIx_ = 0;

  //! increment or decrement and clamp activeSceneInstanceIx_ and load the
  //! corresponding scene instance
  void nextSceneInstance(bool previous = false);

  //! load a particular light setup by handle and register as the default
  //! LightSetup
  void instanceLightSetup(std::string lightSetupHandle);

  bool debugBullet_ = false;

  esp::scene::SceneNode* agentBodyNode_ = nullptr;

  const int defaultAgentId_ = 0;
  esp::agent::Agent::ptr defaultAgent_ = nullptr;

  // Scene or stage file to load
  std::string sceneFileName;
  esp::gfx::RenderCamera* renderCamera_ = nullptr;
  esp::scene::SceneGraph* activeSceneGraph_ = nullptr;
  bool drawObjectBBs = false;

  Mn::Timeline timeline_;

  Mn::ImGuiIntegration::Context imgui_{Mn::NoCreate};
  bool showFPS_ = true;

  // NOTE: Mouse + shift is to select object on the screen!!
  void createPickedObjectVisualizer(unsigned int objectId);
  std::unique_ptr<ObjectPickingHelper> objectPickingHelper_;
  // returns the number of visible drawables (meshVisualizer drawables are not
  // included)
};

Viewer::Viewer(const Arguments& arguments)
    : Mn::Platform::Application{
          arguments,
          Configuration{}.setTitle("Viewer").setWindowFlags(
              Configuration::WindowFlag::Resizable),
          GLConfiguration{}
              .setColorBufferSize(Mn::Vector4i(8, 8, 8, 8))
              .setSampleCount(4)} {
  Cr::Utility::Arguments args;
#ifdef CORRADE_TARGET_EMSCRIPTEN
  args.addNamedArgument("scene")
#else
  args.addArgument("scene")
#endif
      .setHelp("scene", "scene/stage file to load")
      .addSkippedPrefix("magnum", "engine-specific options")
      .setGlobalHelp("Displays a 3D scene file provided on command line")
      .addBooleanOption("enable-physics")
      .addBooleanOption("stage-requires-lighting")
      .setHelp("stage-requires-lighting",
               "Stage asset should be lit with Phong shading.")
      .addBooleanOption("debug-bullet")
      .setHelp("debug-bullet", "Render Bullet physics debug wireframes.")
      .addOption("physics-config", ESP_DEFAULT_PHYSICS_CONFIG_REL_PATH)
      .setHelp("physics-config",
               "Provide a non-default PhysicsManager config file.")
      .addOption("object-dir", "./data/objects")
      .setHelp("object-dir",
               "Provide a directory to search for object config files "
               "(relative to habitat-sim directory).")
      .addBooleanOption("disable-navmesh")
      .setHelp("disable-navmesh",
               "Disable the navmesh, disabling agent navigation constraints.")
      .addOption("navmesh-file")
      .setHelp("navmesh-file", "Manual override path to scene navmesh file.")
      .addBooleanOption("recompute-navmesh")
      .setHelp("recompute-navmesh",
               "Programmatically re-generate the scene navmesh.")
      .addOption("dataset-file")
      .setHelp("dataset-file", "Load a dataset into the MM as primary.")

      .parse(arguments.argc, arguments.argv);

  const auto viewportSize = Mn::GL::defaultFramebuffer.viewport().size();

  imgui_ =
      Mn::ImGuiIntegration::Context(Mn::Vector2{windowSize()} / dpiScaling(),
                                    windowSize(), framebufferSize());

  /* Set up proper blending to be used by ImGui. There's a great chance
     you'll need this exact behavior for the rest of your scene. If not, set
     this only for the drawFrame() call. */
  Mn::GL::Renderer::setBlendEquation(Mn::GL::Renderer::BlendEquation::Add,
                                     Mn::GL::Renderer::BlendEquation::Add);
  Mn::GL::Renderer::setBlendFunction(
      Mn::GL::Renderer::BlendFunction::SourceAlpha,
      Mn::GL::Renderer::BlendFunction::OneMinusSourceAlpha);

  // Setup renderer and shader defaults
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);

  sceneFileName = args.value("scene");
  bool useBullet = args.isSet("enable-physics");
  if (useBullet && (args.isSet("debug-bullet"))) {
    debugBullet_ = true;
  }

  // configure and intialize Simulator
  simConfig_ = esp::sim::SimulatorConfiguration();
  simConfig_.scene.id = sceneFileName;
  simConfig_.enablePhysics = useBullet;
  simConfig_.frustumCulling = true;
  simConfig_.requiresTextures = true;
  if (args.isSet("stage-requires-lighting")) {
    Mn::Debug{} << "Stage using DEFAULT_LIGHTING_KEY";
    simConfig_.sceneLightSetup =
        esp::assets::ResourceManager::DEFAULT_LIGHTING_KEY;
  }

  // setup the PhysicsManager config file
  std::string physicsConfig = Cr::Utility::Directory::join(
      Corrade::Utility::Directory::current(), args.value("physics-config"));
  if (Cr::Utility::Directory::exists(physicsConfig)) {
    Mn::Debug{} << "Using PhysicsManager config: " << physicsConfig;
    simConfig_.physicsConfigFile = physicsConfig;
  }

  // load a dataset and set to active before continuing
  bool loadingFromDatasetConfig = false;
  if (!args.value("dataset-file").empty()) {
    Mn::Debug{} << "~~~~~~~~~~~~~~~ before dataset load ~~~~~~~~~~~~~~~";
    std::string datasetFile = Cr::Utility::Directory::join(
        Corrade::Utility::Directory::current(), args.value("dataset-file"));
    if (Cr::Utility::Directory::exists(datasetFile)) {
      simConfig_.sceneDatasetConfigFile = datasetFile;
      loadingFromDatasetConfig = true;
      // if successfully loaded, we'll do some specific setup next.
      // loadingFromDatasetConfig =
      // simulator_->setActiveDatasetName(datasetFile);
    }
  }

  simulator_ = esp::sim::Simulator::create_unique(simConfig_);

  objectAttrManager_ = simulator_->getObjectAttributesManager();
  objectAttrManager_->loadAllConfigsFromPath(Cr::Utility::Directory::join(
      Corrade::Utility::Directory::current(), args.value("object-dir")));
  assetAttrManager_ = simulator_->getAssetAttributesManager();
  stageAttrManager_ = simulator_->getStageAttributesManager();
  physAttrManager_ = simulator_->getPhysicsAttributesManager();
  sceneAttrManager_ = simulator_->getSceneAttributesManager();

  // manual prototype of scene instance loading from a dataset.
  if (loadingFromDatasetConfig) {
    Mn::Debug{} << "Tried loading a dataset from: "
                << simConfig_.sceneDatasetConfigFile;
    Mn::Debug{} << " - Current dataset: "
                << simulator_->getActiveSceneDatasetName();
    Mn::Debug{} << "    - Current stages available: "
                << stageAttrManager_->getObjectHandlesBySubstring();
    Mn::Debug{} << "    - Current objects available: "
                << objectAttrManager_->getObjectHandlesBySubstring();
    Mn::Debug{} << "    - Current scenes available: "
                << sceneAttrManager_->getObjectHandlesBySubstring();

    // Knowing I am demoing ReplicaCAD, I'll start with empty stage
    loadSceneInstance("empty");
  }

  // NavMesh customization options
  if (args.isSet("disable-navmesh")) {
    if (simulator_->getPathFinder()->isLoaded()) {
      simulator_->setPathFinder(esp::nav::PathFinder::create());
    }
  } else if (args.isSet("recompute-navmesh")) {
    esp::nav::NavMeshSettings navMeshSettings;
    simulator_->recomputeNavMesh(*simulator_->getPathFinder().get(),
                                 navMeshSettings, true);
  } else if (!args.value("navmesh-file").empty()) {
    std::string navmeshFile = Cr::Utility::Directory::join(
        Corrade::Utility::Directory::current(), args.value("navmesh-file"));
    if (Cr::Utility::Directory::exists(navmeshFile)) {
      simulator_->getPathFinder()->loadNavMesh(navmeshFile);
    }
  }

  // configure and initialize default Agent and Sensor
  auto agentConfig = esp::agent::AgentConfiguration();
  agentConfig.height = rgbSensorHeight;
  agentConfig.actionSpace = {
      // setup viewer action space
      {"moveForward",
       esp::agent::ActionSpec::create(
           "moveForward",
           esp::agent::ActuationMap{{"amount", moveSensitivity}})},
      {"moveBackward",
       esp::agent::ActionSpec::create(
           "moveBackward",
           esp::agent::ActuationMap{{"amount", moveSensitivity}})},
      {"moveLeft",
       esp::agent::ActionSpec::create(
           "moveLeft", esp::agent::ActuationMap{{"amount", moveSensitivity}})},
      {"moveRight",
       esp::agent::ActionSpec::create(
           "moveRight", esp::agent::ActuationMap{{"amount", moveSensitivity}})},
      {"moveDown",
       esp::agent::ActionSpec::create(
           "moveDown", esp::agent::ActuationMap{{"amount", moveSensitivity}})},
      {"moveUp",
       esp::agent::ActionSpec::create(
           "moveUp", esp::agent::ActuationMap{{"amount", moveSensitivity}})},
      {"turnLeft",
       esp::agent::ActionSpec::create(
           "turnLeft", esp::agent::ActuationMap{{"amount", lookSensitivity}})},
      {"turnRight",
       esp::agent::ActionSpec::create(
           "turnRight", esp::agent::ActuationMap{{"amount", lookSensitivity}})},
      {"lookUp",
       esp::agent::ActionSpec::create(
           "lookUp", esp::agent::ActuationMap{{"amount", lookSensitivity}})},
      {"lookDown",
       esp::agent::ActionSpec::create(
           "lookDown", esp::agent::ActuationMap{{"amount", lookSensitivity}})},
  };
  agentConfig.sensorSpecifications[0]->resolution =
      esp::vec2i(viewportSize[1], viewportSize[0]);
  // add selects a random initial state and sets up the default controls and
  // step filter
  simulator_->addAgent(agentConfig);

  // Set up camera
  activeSceneGraph_ = &simulator_->getActiveSceneGraph();
  renderCamera_ = &activeSceneGraph_->getDefaultRenderCamera();
  renderCamera_->setAspectRatioPolicy(
      Mn::SceneGraph::AspectRatioPolicy::Extend);
  defaultAgent_ = simulator_->getAgent(defaultAgentId_);
  agentBodyNode_ = &defaultAgent_->node();

  objectPickingHelper_ = std::make_unique<ObjectPickingHelper>(viewportSize);
  timeline_.start();

  printHelpText();
}  // end Viewer::Viewer

int Viewer::addObject(int ID) {
  const std::string& configHandle =
      simulator_->getObjectAttributesManager()->getObjectHandleByID(ID);
  return addObject(configHandle);
}  // addObject

int Viewer::addObject(const std::string& objectAttrHandle) {
  // Relative to agent bodynode
  Mn::Matrix4 T = agentBodyNode_->MagnumObject::transformationMatrix();
  Mn::Vector3 new_pos = T.transformPoint({0.1f, 1.5f, -2.0f});

  int physObjectID = simulator_->addObjectByHandle(objectAttrHandle);
  simulator_->setTranslation(new_pos, physObjectID);
  simulator_->setRotation(Mn::Quaternion::fromMatrix(T.rotationNormalized()),
                          physObjectID);
  return physObjectID;
}  // addObject

// add file-based template derived object from keypress
int Viewer::addTemplateObject() {
  int numObjTemplates = objectAttrManager_->getNumFileTemplateObjects();
  if (numObjTemplates > 0) {
    return addObject(objectAttrManager_->getRandomFileTemplateHandle());
  } else {
    LOG(WARNING) << "No objects loaded, can't add any";
    return esp::ID_UNDEFINED;
  }
}  // addTemplateObject

// add synthesized primiitive object from keypress
int Viewer::addPrimitiveObject() {
  // TODO : use this to implement synthesizing rendered physical objects

  int numObjPrims = objectAttrManager_->getNumSynthTemplateObjects();
  if (numObjPrims > 0) {
    return addObject(objectAttrManager_->getRandomSynthTemplateHandle());
  } else {
    LOG(WARNING) << "No primitive templates available, can't add any objects";
    return esp::ID_UNDEFINED;
  }
}  // addPrimitiveObject

void Viewer::removeLastObject() {
  auto existingObjectIDs = simulator_->getExistingObjectIDs();
  if (existingObjectIDs.size() == 0) {
    return;
  }
  simulator_->removeObject(existingObjectIDs.back());
}

int Viewer::throwSphere(Mn::Vector3 direction) {
  Mn::Matrix4 T =
      agentBodyNode_
          ->MagnumObject::transformationMatrix();  // Relative to agent bodynode

  auto new_pos = Mn::Vector3(defaultAgent_->getSensorSuite()
                                 .get("rgba_camera")
                                 ->specification()
                                 ->position);
  new_pos = T.transformPoint(new_pos);

  std::string sphere_handle =
      simulator_->getObjectAttributesManager()
          ->getSynthTemplateHandlesBySubstring("uvSphereSolid")[0];
  int physObjectID = simulator_->addObjectByHandle(sphere_handle);
  simulator_->setTranslation(new_pos, physObjectID);

  // throw the object
  Mn::Vector3 impulse = direction * 10;
  Mn::Vector3 rel_pos = Mn::Vector3(0.0f, 0.0f, 0.0f);
  simulator_->applyImpulse(impulse, rel_pos, physObjectID);
  return physObjectID;
}

void Viewer::invertGravity() {
  const Mn::Vector3& gravity = simulator_->getGravity();
  const Mn::Vector3 invGravity = -1 * gravity;
  simulator_->setGravity(invGravity);
}

void Viewer::pokeLastObject() {
  auto existingObjectIDs = simulator_->getExistingObjectIDs();
  if (existingObjectIDs.size() == 0)
    return;
  Mn::Matrix4 T =
      agentBodyNode_->MagnumObject::transformationMatrix();  // Relative to
                                                             // agent bodynode
  Mn::Vector3 impulse = T.transformVector({0.0f, 0.0f, -3.0f});
  Mn::Vector3 rel_pos = Mn::Vector3(0.0f, 0.0f, 0.0f);

  simulator_->applyImpulse(impulse, rel_pos, existingObjectIDs.back());
}

void Viewer::pushLastObject() {
  auto existingObjectIDs = simulator_->getExistingObjectIDs();
  if (existingObjectIDs.size() == 0)
    return;
  Mn::Matrix4 T =
      agentBodyNode_->MagnumObject::transformationMatrix();  // Relative to
                                                             // agent bodynode
  Mn::Vector3 force = T.transformVector({0.0f, 0.0f, -40.0f});
  Mn::Vector3 rel_pos = Mn::Vector3(0.0f, 0.0f, 0.0f);
  simulator_->applyForce(force, rel_pos, existingObjectIDs.back());
}

void Viewer::torqueLastObject() {
  auto existingObjectIDs = simulator_->getExistingObjectIDs();
  if (existingObjectIDs.size() == 0)
    return;
  Mn::Vector3 torque = randomDirection() * 30;
  simulator_->applyTorque(torque, existingObjectIDs.back());
}

// generate random direction vectors:
Mn::Vector3 Viewer::randomDirection() {
  Mn::Vector3 dir(1.0f, 1.0f, 1.0f);
  while (sqrt(dir.dot()) > 1.0) {
    dir = Mn::Vector3((float)((rand() % 2000 - 1000) / 1000.0),
                      (float)((rand() % 2000 - 1000) / 1000.0),
                      (float)((rand() % 2000 - 1000) / 1000.0));
  }
  dir = dir / sqrt(dir.dot());
  return dir;
}

void Viewer::wiggleLastObject() {
  // demo of kinematic motion capability
  // randomly translate last added object
  auto existingObjectIDs = simulator_->getExistingObjectIDs();
  if (existingObjectIDs.size() == 0)
    return;

  Mn::Vector3 randDir = randomDirection();
  // Only allow +Y so dynamic objects don't push through the floor.
  randDir[1] = abs(randDir[1]);

  auto translation = simulator_->getTranslation(existingObjectIDs.back());

  simulator_->setTranslation(translation + randDir * 0.1,
                             existingObjectIDs.back());
}

float timeSinceLastSimulation = 0.0;
void Viewer::drawEvent() {
  Mn::GL::defaultFramebuffer.clear(Mn::GL::FramebufferClear::Color |
                                   Mn::GL::FramebufferClear::Depth);

  // step physics at a fixed rate
  timeSinceLastSimulation += timeline_.previousFrameDuration();
  if (timeSinceLastSimulation >= 1.0 / 60.0 &&
      (simulating_ || simulateSingleStep_)) {
    if (mouseGrabber_ != nullptr) {
      auto kinMouseGrabber =
          dynamic_cast<MouseObjectKinematicGrabber*>(mouseGrabber_.get());
      if (kinMouseGrabber != nullptr) {
        // kinMouseGrabber->updateVelAndCache(simStepSize);
      }
    }
    simulator_->stepWorld(1.0 / 60.0);
    timeSinceLastSimulation = 0.0;
    simulateSingleStep_ = false;
  }

  // using polygon offset to increase mesh depth to a avoid z-fighting with
  // debug draw (since lines will not respond to offset).
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::PolygonOffsetFill);
  Mn::GL::Renderer::setPolygonOffset(1.0f, 0.1f);

  // ONLY draw the content to the frame buffer but not immediately blit the
  // result to the default main buffer
  // (this is the reason we do not call displayObservation)
  simulator_->drawObservation(defaultAgentId_, "rgba_camera");
  // TODO: enable other sensors to be displayed

  Mn::GL::Renderer::setDepthFunction(
      Mn::GL::Renderer::DepthFunction::LessOrEqual);
  if (debugBullet_) {
    Mn::Matrix4 camM(renderCamera_->cameraMatrix());
    Mn::Matrix4 projM(renderCamera_->projectionMatrix());

    simulator_->physicsDebugDraw(projM * camM);
  }
  Mn::GL::Renderer::setDepthFunction(Mn::GL::Renderer::DepthFunction::Less);
  Mn::GL::Renderer::setPolygonOffset(0.0f, 0.0f);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::PolygonOffsetFill);

  uint32_t visibles = renderCamera_->getPreviousNumVisibileDrawables();

  esp::gfx::RenderTarget* sensorRenderTarget =
      simulator_->getRenderTarget(defaultAgentId_, "rgba_camera");
  CORRADE_ASSERT(sensorRenderTarget,
                 "Error in Viewer::drawEvent: sensor's rendering target "
                 "cannot be nullptr.", );
  if (objectPickingHelper_->isObjectPicked()) {
    // we need to immediately draw picked object to the SAME frame buffer
    // so bind it first
    // bind the framebuffer
    sensorRenderTarget->renderReEnter();

    // setup blending function
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::Blending);

    // render the picked object on top of the existing contents
    esp::gfx::RenderCamera::Flags flags;
    if (simulator_->isFrustumCullingEnabled()) {
      flags |= esp::gfx::RenderCamera::Flag::FrustumCulling;
    }
    renderCamera_->draw(objectPickingHelper_->getDrawables(), flags);

    Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::Blending);
  }

  sensorRenderTarget->blitRgbaToDefault();
  // Immediately bind the main buffer back so that the "imgui" below can work
  // properly
  Mn::GL::defaultFramebuffer.bind();

  imgui_.newFrame();
  // Show mouseInteractionMode
  ImGui::SetNextWindowPos(ImVec2(10, 10));
  ImGui::Begin("main", NULL,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                   ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::SetWindowFontScale(2.0);
  std::string mouseModeText =
      "Mouse Mode: " + getEnumName(mouseInteractionMode);
  ImGui::Text(mouseModeText.c_str());

  if (showFPS_) {
    ImGui::Text("%.1f FPS", Mn::Double(ImGui::GetIO().Framerate));
    uint32_t total = activeSceneGraph_->getDrawables().size();
    ImGui::Text("%u drawables", total);
    ImGui::Text("%u culled", total - visibles);
  }

  ImGui::End();

  /* Set appropriate states. If you only draw ImGui, it is sufficient to
     just enable blending and scissor test in the constructor. */
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::Blending);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::ScissorTest);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::FaceCulling);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::DepthTest);

  imgui_.drawFrame();

  /* Reset state. Only needed if you want to draw something else with
     different state after. */

  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::ScissorTest);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::Blending);

  swapBuffers();
  timeline_.nextFrame();
  redraw();
}

void Viewer::viewportEvent(ViewportEvent& event) {
  auto& sensors = defaultAgent_->getSensorSuite();
  for (auto entry : sensors.getSensors()) {
    auto visualSensor =
        dynamic_cast<esp::sensor::VisualSensor*>(entry.second.get());
    if (visualSensor != nullptr) {
      visualSensor->specification()->resolution = {event.windowSize()[1],
                                                   event.windowSize()[0]};
      simulator_->getRenderer()->bindRenderTarget(*visualSensor);
    }
  }
  Mn::GL::defaultFramebuffer.setViewport({{}, framebufferSize()});

  imgui_.relayout(Mn::Vector2{event.windowSize()} / event.dpiScaling(),
                  event.windowSize(), event.framebufferSize());

  objectPickingHelper_->handleViewportChange(event.framebufferSize());
}

void Viewer::createPickedObjectVisualizer(unsigned int objectId) {
  for (auto& it : activeSceneGraph_->getDrawableGroups()) {
    if (it.second.hasDrawable(objectId)) {
      auto* pickedDrawable = it.second.getDrawable(objectId);
      objectPickingHelper_->createPickedObjectVisualizer(pickedDrawable);
      break;
    }
  }
}

void Viewer::mousePressEvent(MouseEvent& event) {
  event.setAccepted();
  if (event.button() == MouseEvent::Button::Right &&
      (event.modifiers() & MouseEvent::Modifier::Shift)) {
    // cannot use the default framebuffer, so setup another framebuffer,
    // also, setup the color attachment for rendering, and remove the visualizer
    // for the previously picked object
    objectPickingHelper_->prepareToDraw();

    // redraw the scene on the object picking framebuffer
    esp::gfx::RenderCamera::Flags flags =
        esp::gfx::RenderCamera::Flag::UseDrawableIdAsObjectId;
    if (simulator_->isFrustumCullingEnabled())
      flags |= esp::gfx::RenderCamera::Flag::FrustumCulling;
    for (auto& it : activeSceneGraph_->getDrawableGroups()) {
      renderCamera_->draw(it.second, flags);
    }

    // Read the object Id
    unsigned int pickedObject =
        objectPickingHelper_->getObjectId(event.position(), windowSize());

    // if an object is selected, create a visualizer
    createPickedObjectVisualizer(pickedObject);
    return;
  }  // drawable selection

  auto viewportPoint = event.position();
  auto ray = renderCamera_->unproject(viewportPoint);
  if (mouseInteractionMode == THROW) {
    throwSphere(ray.direction);
    return;
  }
  esp::physics::RaycastResults raycastResults = simulator_->castRay(ray);
  if (raycastResults.hasHits()) {
    auto hitInfo = raycastResults.hits[0];
    if (mouseInteractionMode == ADD) {
      if (event.button() == MouseEvent::Button::Left) {
        // add primitive w/ left click if a collision object is hit by a raycast
        // NOTE: right click and drag to select a object for placement
        int objID = addPrimitiveObject();
        // use the bounding box to create a safety margin for adding the object
        float boundingBuffer = simulator_->getObjectSceneNode(objID)
                                       ->computeCumulativeBB()
                                       .size()
                                       .max() /
                                   2.0 +
                               0.04;
        simulator_->setTranslation(
            hitInfo.point + hitInfo.normal * boundingBuffer, objID);

        simulator_->setRotation(esp::core::randomRotation(), objID);
      }  // end add primitive w/ left click
    } else if (mouseInteractionMode == REMOVE) {
      if (hitInfo.objectId >= 0) {
        simulator_->removeObject(hitInfo.objectId);
      }
    } else if (mouseInteractionMode == GRAB) {
      if (hitInfo.objectId >= 0) {
        mouseGrabber_ = std::make_unique<MouseObjectKinematicGrabber>(
            hitInfo.point,
            (hitInfo.point - renderCamera_->node().translation()).length(),
            hitInfo.objectId, simulator_.get());
        if (event.button() == MouseEvent::Button::Right) {
          mouseGrabber_->mode = 1;
        }
      }
    }
  }
}

void Viewer::mouseReleaseEvent(MouseEvent& event) {
  if (mouseInteractionMode == GRAB) {
    mouseGrabber_ = nullptr;
  }
  event.setAccepted();
}

void Viewer::mouseScrollEvent(MouseScrollEvent& event) {
  if (!event.offset().y()) {
    return;
  }

  if (mouseGrabber_ != nullptr) {
    if (mouseGrabber_->mode == 0) {  // LEFT click
      // adjust the grabber depth
      auto ray = renderCamera_->unproject(event.position());
      mouseGrabber_->gripDepth += event.offset().y() * 0.01;
      mouseGrabber_->targetFrame.translation =
          renderCamera_->node().absoluteTranslation() +
          ray.direction * mouseGrabber_->gripDepth;
      mouseGrabber_->updateTarget(mouseGrabber_->targetFrame);
    } else if (mouseGrabber_->mode == 1) {  // RIGHT click
      // roll the object
      auto roll_quat = Mn::Quaternion::rotation(
          Mn::Deg(event.offset().y()),
          defaultAgent_->node().transformation().transformVector({0, 0, -1.0}));
      mouseGrabber_->targetFrame.rotation =
          roll_quat * mouseGrabber_->targetFrame.rotation;
      mouseGrabber_->updateTarget(mouseGrabber_->targetFrame);
    }
  } else {
    // change the mouse interaction mode
    int delta = 1;
    if (event.offset().y() < 0)
      delta = -1;
    mouseInteractionMode = MouseInteractionMode(
        (int(mouseInteractionMode) + delta) % int(NUM_MODES));
    if (mouseInteractionMode < 0)
      mouseInteractionMode = MouseInteractionMode(int(NUM_MODES) - 1);
  }

  redraw();

  event.setAccepted();
}

void Viewer::mouseMoveEvent(MouseMoveEvent& event) {
  const Mn::Vector2i delta = event.relativePosition();

  if (mouseGrabber_ != nullptr) {
    if (event.buttons() & MouseMoveEvent::Button::Left) {
      auto ray = renderCamera_->unproject(event.position());
      mouseGrabber_->targetFrame.translation =
          renderCamera_->node().absoluteTranslation() +
          ray.direction * mouseGrabber_->gripDepth;
      mouseGrabber_->updateTarget(mouseGrabber_->targetFrame);
    } else if (event.buttons() & MouseMoveEvent::Button::Right) {
      auto y_quat = Mn::Quaternion::rotation(Mn::Deg(delta.x()), {0, 1.0, 0});
      auto x_quat = Mn::Quaternion::rotation(
          Mn::Deg(delta.y()),
          defaultAgent_->node().transformation().transformVector({1.0, 0, 0}));
      mouseGrabber_->targetFrame.rotation =
          x_quat * y_quat * mouseGrabber_->targetFrame.rotation;
      mouseGrabber_->updateTarget(mouseGrabber_->targetFrame);
    }
  }

  if (mouseInteractionMode == LOOK &&
      (event.buttons() & MouseMoveEvent::Button::Left)) {
    auto& controls = *defaultAgent_->getControls().get();
    controls(*agentBodyNode_, "turnRight", delta.x());
    // apply the transformation to all sensors
    for (auto p : defaultAgent_->getSensorSuite().getSensors()) {
      controls(p.second->object(),  // SceneNode
               "lookDown",          // action name
               delta.y(),           // amount
               false);              // applyFilter
    }
  }

  redraw();

  event.setAccepted();
}

// NOTE: Mouse + shift is to select object on the screen!!
void Viewer::keyPressEvent(KeyEvent& event) {
  const auto key = event.key();
  switch (key) {
    case KeyEvent::Key::Esc:
      std::exit(0);
      break;
    case KeyEvent::Key::Space:
      simulating_ = !simulating_;
      Mn::Debug{} << " Physics Simulation: " << simulating_;
      break;
    case KeyEvent::Key::Period:
      // also `>` key
      simulateSingleStep_ = true;
      break;
    case KeyEvent::Key::Left:
      defaultAgent_->act("turnLeft");
      break;
    case KeyEvent::Key::Right:
      defaultAgent_->act("turnRight");
      break;
    case KeyEvent::Key::Up:
      defaultAgent_->act("lookUp");
      break;
    case KeyEvent::Key::Down:
      defaultAgent_->act("lookDown");
      break;
    case KeyEvent::Key::Eight:
      addPrimitiveObject();
      break;
    case KeyEvent::Key::Nine:
      if (simulator_->getPathFinder()->isLoaded()) {
        const esp::vec3f position =
            simulator_->getPathFinder()->getRandomNavigablePoint();
        agentBodyNode_->setTranslation(Mn::Vector3(position));
      }
      break;
    case KeyEvent::Key::A:
      defaultAgent_->act("moveLeft");
      break;
    case KeyEvent::Key::D:
      defaultAgent_->act("moveRight");
      break;
    case KeyEvent::Key::S:
      defaultAgent_->act("moveBackward");
      break;
    case KeyEvent::Key::W:
      defaultAgent_->act("moveForward");
      break;
    case KeyEvent::Key::X:
      defaultAgent_->act("moveDown");
      break;
    case KeyEvent::Key::Z:
      defaultAgent_->act("moveUp");
      break;
    case KeyEvent::Key::E:
      simulator_->setFrustumCullingEnabled(
          !simulator_->isFrustumCullingEnabled());
      break;
    case KeyEvent::Key::C:
      showFPS_ = !showFPS_;
      break;
    case KeyEvent::Key::O:
      addTemplateObject();
      break;
    case KeyEvent::Key::P:
      pokeLastObject();
      break;
    case KeyEvent::Key::F:
      pushLastObject();
      break;
    case KeyEvent::Key::K:
      wiggleLastObject();
      break;
    case KeyEvent::Key::U:
      removeLastObject();
      break;
    case KeyEvent::Key::V:
      invertGravity();
      break;
    case KeyEvent::Key::T:
      // Test key. Put what you want here...
      torqueLastObject();
      break;
    case KeyEvent::Key::N:
      // toggle navmesh visualization
      simulator_->setNavMeshVisualization(
          !simulator_->isNavMeshVisualizationActive());
      break;
    case KeyEvent::Key::I:
      screenshot();
      break;
    case KeyEvent::Key::Q:
      // query the agent state
      logAgentStateMsg(true, true);
      break;
    case KeyEvent::Key::B: {
      // toggle bounding box on objects
      drawObjectBBs = !drawObjectBBs;
      for (auto id : simulator_->getExistingObjectIDs()) {
        simulator_->setObjectBBDraw(drawObjectBBs, id);
      }
    } break;
    case KeyEvent::Key::H:
      printHelpText();
      break;
    case KeyEvent::Key::LeftBracket: {
      nextSceneInstance(true);  // previous
    } break;
    case KeyEvent::Key::RightBracket: {
      nextSceneInstance();  // next
    } break;
    default:
      break;
  }
  redraw();
}

int savedFrames = 0;
//! Save a screenshot to "screenshots/year_month_day_hour-minute-second/#.png"
void Viewer::screenshot() {
  std::string screenshot_directory =
      "screenshots/" + viewerStartTimeString + "/";
  if (!Cr::Utility::Directory::exists(screenshot_directory)) {
    Cr::Utility::Directory::mkpath(screenshot_directory);
  }
  Mn::DebugTools::screenshot(
      Mn::GL::defaultFramebuffer,
      screenshot_directory + std::to_string(savedFrames++) + ".png");
}

void Viewer::nextSceneInstance(bool previous) {
  auto agentState = esp::agent::AgentState::create();
  simulator_->getAgent(0)->getState(agentState);
  activeSceneInstanceIx_ =
      std::min(sceneAttrManager_->getNumObjects() - 1,
               std::max(activeSceneInstanceIx_ + (previous ? -1 : 1), 0));
  if (activeSceneInstanceIx_ >= 0) {
    loadSceneInstance(
        sceneAttrManager_
            ->getObjectHandlesBySubstring()[activeSceneInstanceIx_]);
  }
  simulator_->getAgent(0)->setState(*agentState.get());
  Mn::Debug{} << "Loaded instance " << activeSceneInstanceIx_ << " : "
              << sceneAttrManager_
                     ->getObjectHandlesBySubstring()[activeSceneInstanceIx_];
}

// clear the scene and then attempt to manually load a scene instance
void Viewer::loadSceneInstance(std::string sceneInstanceHandle) {
  Mn::Debug{} << "Viewer::loadSceneInstance(" << sceneInstanceHandle << "):";
  Mn::Debug{} << "Active dataset name = "
              << simulator_->getActiveSceneDatasetName();
  auto matchingSceneHandles =
      sceneAttrManager_->getObjectHandlesBySubstring(sceneInstanceHandle);
  Mn::Debug{} << "  - matching scene handles: " << matchingSceneHandles;

  auto sceneTemplate =
      sceneAttrManager_->getObjectByHandle(matchingSceneHandles[0]);

  Mn::Debug{} << "  - got stage instance: "
              << sceneTemplate->getStageInstance();

  // 1. reconfigure with new stage
  // note: opportunity for ambiguity
  simConfig_.scene.id = stageAttrManager_->getObjectHandlesBySubstring(
      sceneTemplate->getStageInstance()->getHandle())[0];
  if (stageAttrManager_->getObjectByHandle(simConfig_.scene.id)
          ->getRequiresLighting()) {
    simConfig_.sceneLightSetup =
        esp::assets::ResourceManager::DEFAULT_LIGHTING_KEY;
  } else {
    simConfig_.sceneLightSetup = esp::assets::ResourceManager::NO_LIGHT_KEY;
  }
  // TODO: lighting
  Mn::Debug{} << "  - stage handle: " << simConfig_.scene.id;

  simulator_->reconfigure(simConfig_);
  // remove any objects leftover (when stage did not change)
  for (auto id : simulator_->getExistingObjectIDs()) {
    simulator_->removeObject(id);
  }

  // 2. load new NavMesh
  Mn::Debug{} << "Load navmesh: " << sceneTemplate->getNavmeshHandle();
  simulator_->getPathFinder().reset();
  if (!sceneTemplate->getNavmeshHandle().empty()) {
    if (simulator_->getMetadataMediator()->getActiveNavmeshMap().count(
            sceneTemplate->getNavmeshHandle()) > 0) {
      auto navmeshSource =
          simulator_->getMetadataMediator()->getActiveNavmeshMap().at(
              sceneTemplate->getNavmeshHandle());
      auto navmeshFullSource = Cr::Utility::Directory::join(
          Cr::Utility::Directory::path(simConfig_.sceneDatasetConfigFile),
          navmeshSource);

      Mn::Debug{} << " attemping to load " << navmeshSource;
      Mn::Debug{} << "    abs path: " << navmeshFullSource;
      bool success =
          simulator_->getPathFinder()->loadNavMesh(navmeshFullSource);
      Mn::Debug{} << " ... success : " << success;
    } else {
      Mn::Debug{} << " ... configured handle is not valid, aborting.";
    }
  }
  // TODO: this should be more graceful
  if (simulator_->isNavMeshVisualizationActive()) {
    simulator_->setNavMeshVisualization(false);
    simulator_->setNavMeshVisualization(true);
  }

  // 3. load the LightSetup
  if (!sceneTemplate->getLightingHandle().empty()) {
    instanceLightSetup(sceneTemplate->getLightingHandle());
  }

  // 4. load new objects
  for (auto objectInstance : sceneTemplate->getObjectInstances()) {
    // note: opportunity for ambiguity due to substring search
    auto objectHandle = objectAttrManager_->getObjectHandlesBySubstring(
        objectInstance->getHandle())[0];
    auto id = simulator_->addObjectByHandle(objectHandle);
    auto objCOMShift =
        simulator_->getObjectVisualSceneNodes(id)[0]->translation();
    simulator_->setTranslation(objectInstance->getTranslation() - objCOMShift,
                               id);
    simulator_->setRotation(objectInstance->getRotation(), id);
    simulator_->setObjectMotionType(
        esp::physics::MotionType(objectInstance->getMotionType()), id);
    Mn::Debug{} << "======================";
    Mn::Debug{} << " - Looking for object " << objectInstance->getHandle()
                << "    ... found object handle : " << objectHandle;
    Mn::Debug{} << " - MotionType = " << objectInstance->getMotionType();
  }
}

void Viewer::instanceLightSetup(std::string lightSetupHandle) {
  Mn::Debug{} << "Viewer::instanceLightSetup(" << lightSetupHandle << ")";
  auto lightMgr =
      simulator_->getMetadataMediator()->getLightLayoutAttributesManager();
  auto matchingHandles =
      lightMgr->getObjectHandlesBySubstring(lightSetupHandle);
  if (!matchingHandles.empty()) {
    Mn::Debug{} << "  ... handle(s) found: " << matchingHandles;
    Mn::Debug{} << "  ... creating LightSetup from " << matchingHandles[0];
    auto lightLayoutAttr = lightMgr->getObjectByHandle(matchingHandles[0]);
    esp::gfx::LightSetup lightSetup;

    for (auto lightInstanceInfo : lightLayoutAttr->getLightInstances()) {
      esp::gfx::LightInfo lightInfo;
      lightInfo.color = lightInstanceInfo.second->getColor() *
                        lightInstanceInfo.second->getIntensity();
      if (lightInstanceInfo.second->getType() == "point") {
        lightInfo.vector = {lightInstanceInfo.second->getPosition(), 1};
      } else if (lightInstanceInfo.second->getType() == "directional") {
        lightInfo.vector = {lightInstanceInfo.second->getDirection(), 0};
      } else {
        Mn::Debug{} << " light with type "
                    << lightInstanceInfo.second->getType()
                    << " not supported. Skipping.";
        continue;
      }
      lightSetup.push_back(lightInfo);
    }
    // override the default light setup for simplicity
    simulator_->setLightSetup(lightSetup);
  } else {
    Mn::Debug{} << "  ... handle not found, aborting.";
  }
}

}  // namespace

MAGNUM_APPLICATION_MAIN(Viewer)
