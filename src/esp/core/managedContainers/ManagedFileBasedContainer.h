// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_CORE_MANAGEDFILEBASEDCONTAINER_H_
#define ESP_CORE_MANAGEDFILEBASEDCONTAINER_H_

/** @file
 * @brief Class Template @ref esp::core::ManagedFileBasedContainer : @ref
 * esp::core::ManagedContainer functionality specifically for file-based @ref
 * esp::core::AbstractManagedObject objects
 */

#include "ManagedContainer.h"

#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>

#include "esp/io/io.h"
#include "esp/io/json.h"

namespace esp {
namespace core {
/**
 * @brief Class template defining file-io-based responsibilities and
 * functionality for managing @ref esp::core::AbstractFileBasedManagedObject
 * constructs.
 * @tparam T the type of managed object a particular specialization of
 * this class works with.  Must inherit from @ref
 * esp::core::AbstractFileBasedManagedObject.
 * @tparam Access Whether the default access (getters) for this
 * container provides copies of the objects held, or the actual objects
 * themselves.
 */
template <class T, ManagedObjectAccess Access>
class ManagedFileBasedContainer : public ManagedContainer<T, Access> {
 public:
  static_assert(std::is_base_of<AbstractFileBasedManagedObject, T>::value,
                "ManagedContainer :: Managed object type must be derived from "
                "AbstractFileBasedManagedObject");
  typedef std::shared_ptr<T> ManagedFileIOPtr;

  explicit ManagedFileBasedContainer(const std::string& metadataType)
      : ManagedContainer<T, Access>(metadataType) {}

  /**
   * @brief Utility function to check if passed string represents an existing,
   * user-accessible file
   * @param handle the string to check
   * @return whether the file exists in the file system and whether the user has
   * access
   */
  bool isValidFileName(const std::string& handle) const {
    return (Corrade::Utility::Directory::exists(handle));
  }  // ManagedFileBasedContainer::isValidFileName

  /**
   * @brief Creates an instance of a managed object from a JSON file.
   *
   * @param filename the name of the file describing the object managed object.
   * Assumes it exists and fails if it does not.
   * @param registerObject whether to add this managed object to the
   * library. If the user is going to edit this managed object, this should be
   * false - any subsequent editing will require re-registration. Defaults to
   * true.
   * @return a reference to the desired managed object, or nullptr if fails.
   */
  ManagedFileIOPtr createObjectFromJSONFile(const std::string& filename,
                                            bool registerObject = true) {
    io::JsonDocument docConfig = nullptr;
    bool success = this->verifyLoadDocument(filename, docConfig);
    if (!success) {
      LOG(ERROR) << "ManagedFileBasedContainer::createObjectFromFile ("
                 << this->objectType_
                 << ") : Failure reading document as JSON : " << filename
                 << ". Aborting.";
      return nullptr;
    }
    // convert doc to const value
    const io::JsonGenericValue config = docConfig.GetObject();
    ManagedFileIOPtr attr = this->buildManagedObjectFromDoc(filename, config);
    return this->postCreateRegister(attr, registerObject);
  }  // ManagedFileBasedContainer::createObjectFromJSONFile

  /**
   * @brief Method to load a Managed Object's data from a file.  If the file
   * type is not supported by specialization of this method, this method
   * executes and an error is thrown.
   * @tparam type of document to load.
   * @param filename name of file document to load from
   * @param config document to read for data
   * @return a shared pointer of the created Managed Object
   */
  template <typename U>
  ManagedFileIOPtr buildManagedObjectFromDoc(const std::string& filename,
                                             CORRADE_UNUSED const U& config) {
    LOG(ERROR)
        << "ManagedContainer::buildManagedObjectFromDoc (" << this->objectType_
        << ") : Failure loading attributes from document of unknown type : "
        << filename << ". Aborting.";
  }
  /**
   * @brief Method to load a Managed Object's data from a file.  This is the
   * JSON specialization, using type inference.
   * @param filename name of file document to load from
   * @param config JSON document to read for data
   * @return a shared pointer of the created Managed Object
   */
  ManagedFileIOPtr buildManagedObjectFromDoc(
      const std::string& filename,
      const io::JsonGenericValue& jsonConfig) {
    return this->buildObjectFromJSONDoc(filename, jsonConfig);
  }

  /**
   * @brief Parse passed JSON Document specifically for @ref ManagedPtr object.
   * It always returns a @ref ManagedPtr object.
   * @param filename The name of the file describing the @ref ManagedPtr,
   * used as managed object handle/name on create.
   * @param jsonConfig json document to parse - assumed to be legal JSON doc.
   * @return a reference to the desired managed object.
   */
  virtual ManagedFileIOPtr buildObjectFromJSONDoc(
      const std::string& filename,
      const io::JsonGenericValue& jsonConfig) = 0;

 protected:
  //======== Common File-based import and utility functions ========

  /**
   * @brief Verify passd @p filename is legal document of type T. Returns loaded
   * document in passed argument if successful. This requires appropriate
   * specialization for each type name, so if this method is executed it means
   * no appropriate specialization exists for passed type of document.
   *
   * @tparam type of document
   * @param filename name of potentia document to load
   * @param resDoc a reference to the document to be parsed.
   * @return whether document has been loaded successfully or not
   */
  template <class U>
  bool verifyLoadDocument(const std::string& filename,
                          CORRADE_UNUSED U& resDoc) {
    // by here always fail
    LOG(ERROR) << this->objectType_
               << "ManagedContainerBase::verifyLoadDocument : File " << filename
               << " failed due to unknown file type.";
    return false;
  }  // ManagedContainerBase::verifyLoadDocument
  /**
   * @brief Verify passed @p filename is legal json document, return loaded
   * document or nullptr if fails
   *
   * @param filename name of potential json document to load
   * @param jsonDoc a reference to the json document to be parsed
   * @return whether document has been loaded successfully or not
   */
  bool verifyLoadDocument(const std::string& filename,
                          io::JsonDocument& jsonDoc);

  /**
   * @brief Will build a json file name for @p filename by appending/replacing
   * the extension with the passed @p jsonTypeExt, if it is missing.  NOTE :
   * this does not verify that file exists.
   * @param filename The original file name
   * @param jsonTypeExt The extension to use.
   * @return The file name changed so that it has the correct @p jsonTypeExtif
   * it was missing.
   */
  std::string convertFilenameToJSON(const std::string& filename,
                                    const std::string& jsonTypeExt);

  /**
   * @brief Get directory component of managed object handle and call @ref
   * esp::core::AbstractManagedObject::setFileDirectory if a legitimate
   * directory exists in handle.
   *
   * @param object pointer to managed object to set
   */
  void setFileDirectoryFromHandle(ManagedFileIOPtr object) {
    std::string handleName = object->getHandle();
    auto loc = handleName.find_last_of('/');
    if (loc != std::string::npos) {
      object->setFileDirectory(handleName.substr(0, loc));
    }
  }  // setFileDirectoryFromHandle

 public:
  ESP_SMART_POINTERS(ManagedFileBasedContainer<T, Access>)

};  // class ManagedFileBasedContainer

/////////////////////////////
// Class Template Method Definitions

template <class T, ManagedObjectAccess Access>

std::string ManagedFileBasedContainer<T, Access>::convertFilenameToJSON(
    const std::string& filename,
    const std::string& jsonTypeExt) {
  std::string strHandle = Cr::Utility::String::lowercase(filename);
  std::string resHandle(filename);
  if (std::string::npos ==
      strHandle.find(Cr::Utility::String::lowercase(jsonTypeExt))) {
    resHandle = Cr::Utility::Directory::splitExtension(filename).first + "." +
                jsonTypeExt;
    LOG(INFO) << "ManagedFileBasedContainer<" << this->objectType_
              << ">::convertFilenameToJSON : Filename : " << filename
              << " changed to proposed JSON configuration filename : "
              << resHandle;
  } else {
    LOG(INFO) << "ManagedFileBasedContainer<" << this->objectType_
              << ">::convertFilenameToJSON : Filename : " << filename
              << " is appropriate JSON configuration filename.";
  }
  return resHandle;
}  // ManagedFileBasedContainer<T, Access>::convertFilenameToJSON

template <class T, ManagedObjectAccess Access>
bool ManagedFileBasedContainer<T, Access>::verifyLoadDocument(
    const std::string& filename,
    io::JsonDocument& jsonDoc) {
  if (isValidFileName(filename)) {
    try {
      jsonDoc = io::parseJsonFile(filename);
    } catch (...) {
      LOG(ERROR) << "ManagedFileBasedContainer<" << this->objectType_
                 << ">::verifyLoadDocument : Failed to parse " << filename
                 << " as JSON.";
      return false;
    }
    return true;
  } else {
    // by here always fail
    LOG(ERROR) << "ManagedFileBasedContainer<" << this->objectType_
               << ">::verifyLoadDocument : File " << filename
               << " does not exist";
    return false;
  }
}  // ManagedFileBasedContainer<T, Access>::verifyLoadDocument

}  // namespace core
}  // namespace esp
#endif  // ESP_CORE_MANAGEDFILEBASEDCONTAINER_H_
