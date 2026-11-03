#include <Saba/Base/Path.h>
#include <Saba/Base/UnicodeUtil.h>
#include <Saba/Model/MMD/MMDModel.h>
#include <Saba/Model/MMD/PMDModel.h>
#include <Saba/Model/MMD/PMXModel.h>
#include <Saba/Model/MMD/VMDAnimation.h>
#include <Saba/Model/MMD/VMDFile.h>

#include <nlohmann/json.hpp>

#include <glm/glm.hpp>

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <shellapi.h>
#endif

namespace
{
using json = nlohmann::json;

struct Options
{
    std::string modelPath;
    std::vector<std::string> vmdPaths;
    std::string outputPath;
    std::string globalTransformPath;
    int startFrame = 0;
    int endFrame = -1;
    int step = 1;
    bool legacyTargetsV1 = false;
};

struct BonePosition
{
    bool found = false;
    glm::vec3 value = glm::vec3(0.0f);
};

struct RegionBounds
{
    bool found = false;
    glm::vec3 min = glm::vec3(0.0f);
    glm::vec3 max = glm::vec3(0.0f);
};

void Usage()
{
    std::cout
        << "mmd_target_dump <pmd/pmx file> -vmd <vmd file> [-vmd <vmd file> ...] -o <output.json>\n"
        << "                [-global-transform <output.json>]\n"
        << "                [-legacy-targets-v1]\n"
        << "                [-start <frame>] [-end <frame>] [-step <frame step>]\n";
}

bool ParseOptions(const std::vector<std::string>& args, Options* options)
{
    if (args.size() <= 1)
    {
        return false;
    }

    options->modelPath = args[1];
    for (size_t i = 2; i < args.size(); i++)
    {
        if (args[i] == "-vmd")
        {
            i++;
            if (i >= args.size())
            {
                return false;
            }
            options->vmdPaths.push_back(args[i]);
        }
        else if (args[i] == "-o")
        {
            i++;
            if (i >= args.size())
            {
                return false;
            }
            options->outputPath = args[i];
        }
        else if (args[i] == "-global-transform")
        {
            i++;
            if (i >= args.size())
            {
                return false;
            }
            options->globalTransformPath = args[i];
        }
        else if (args[i] == "-legacy-targets-v1")
        {
            options->legacyTargetsV1 = true;
        }
        else if (args[i] == "-start")
        {
            i++;
            if (i >= args.size())
            {
                return false;
            }
            options->startFrame = std::stoi(args[i]);
        }
        else if (args[i] == "-end")
        {
            i++;
            if (i >= args.size())
            {
                return false;
            }
            options->endFrame = std::stoi(args[i]);
        }
        else if (args[i] == "-step")
        {
            i++;
            if (i >= args.size())
            {
                return false;
            }
            options->step = (std::max)(1, std::stoi(args[i]));
        }
        else
        {
            return false;
        }
    }

    return !options->modelPath.empty()
        && !options->vmdPaths.empty()
        && (!options->outputPath.empty() || !options->globalTransformPath.empty());
}

std::shared_ptr<saba::MMDModel> LoadModel(const std::string& modelPath)
{
    std::shared_ptr<saba::MMDModel> mmdModel;
    const std::string ext = saba::PathUtil::GetExt(modelPath);
    if (ext == "pmd")
    {
        auto pmdModel = std::make_unique<saba::PMDModel>();
        if (!pmdModel->Load(modelPath, ""))
        {
            return nullptr;
        }
        mmdModel = std::move(pmdModel);
    }
    else if (ext == "pmx")
    {
        auto pmxModel = std::make_unique<saba::PMXModel>();
        if (!pmxModel->Load(modelPath, ""))
        {
            return nullptr;
        }
        mmdModel = std::move(pmxModel);
    }
    return mmdModel;
}

bool LoadAnimation(
    const std::shared_ptr<saba::MMDModel>& model,
    const std::vector<std::string>& vmdPaths,
    std::unique_ptr<saba::VMDAnimation>* outAnim)
{
    auto vmdAnim = std::make_unique<saba::VMDAnimation>();
    if (!vmdAnim->Create(model))
    {
        return false;
    }

    for (const auto& vmdPath : vmdPaths)
    {
        saba::VMDFile vmdFile;
        if (!saba::ReadVMDFile(&vmdFile, vmdPath.c_str()))
        {
            return false;
        }
        if (!vmdAnim->Add(vmdFile))
        {
            return false;
        }
    }

    *outAnim = std::move(vmdAnim);
    return true;
}

glm::vec3 ExtractTranslation(const glm::mat4& m)
{
    return glm::vec3(m[3]);
}

std::vector<std::string> MakeAliases(std::initializer_list<const char*> names)
{
    return std::vector<std::string>(names.begin(), names.end());
}

struct SimplifiedBoneSpec
{
    std::string boneName;
    std::vector<std::string> aliases;
};

std::vector<SimplifiedBoneSpec> MakeSimplifiedBoneSpecs()
{
    return {
        { "lower_body", MakeAliases({ u8"\u4E0B\u534A\u8EAB", "lower body", "LowerBody" }) },
        { "upper_body", MakeAliases({ u8"\u4E0A\u534A\u8EAB", "upper body", "UpperBody" }) },
        { "upper_body2", MakeAliases({ u8"\u4E0A\u534A\u8EAB2", "upper body2", "UpperBody2" }) },
        { "waist", MakeAliases({ u8"\u8170", "waist", "Waist" }) },
        { "head", MakeAliases({ u8"\u982D", "head", "Head" }) },
        { "left_eye", MakeAliases({ u8"\u5DE6\u76EE", "left eye", "LeftEye", "eye_l", "Eye_L" }) },
        { "right_eye", MakeAliases({ u8"\u53F3\u76EE", "right eye", "RightEye", "eye_r", "Eye_R" }) },
        { "both_eyes", MakeAliases({ u8"\u4E21\u76EE", "eyes", "BothEyes" }) },
        { "left_leg", MakeAliases({ u8"\u5DE6\u8DB3", "left leg", "LeftLeg", "leg_l", "Leg_L" }) },
        { "left_knee", MakeAliases({ u8"\u5DE6\u3072\u3056", "left knee", "LeftKnee", "knee_l", "Knee_L" }) },
        { "left_ankle", MakeAliases({ u8"\u5DE6\u8DB3\u9996", "left ankle", "LeftAnkle", "ankle_l", "Ankle_L" }) },
        { "left_toe", MakeAliases({ u8"\u5DE6\u3064\u307E\u5148", "left toe", "LeftToe", "toe_l", "Toe_L" }) },
        { "right_leg", MakeAliases({ u8"\u53F3\u8DB3", "right leg", "RightLeg", "leg_r", "Leg_R" }) },
        { "right_knee", MakeAliases({ u8"\u53F3\u3072\u3056", "right knee", "RightKnee", "knee_r", "Knee_R" }) },
        { "right_ankle", MakeAliases({ u8"\u53F3\u8DB3\u9996", "right ankle", "RightAnkle", "ankle_r", "Ankle_R" }) },
        { "right_toe", MakeAliases({ u8"\u53F3\u3064\u307E\u5148", "right toe", "RightToe", "toe_r", "Toe_R" }) },
        { "left_shoulder_p", MakeAliases({ u8"\u5DE6\u80A9P", "left shoulder p", "LeftShoulderP" }) },
        { "left_shoulder", MakeAliases({ u8"\u5DE6\u80A9", "left shoulder", "LeftShoulder", "shoulder_l", "Shoulder_L" }) },
        { "left_arm", MakeAliases({ u8"\u5DE6\u8155", "left arm", "LeftArm", "arm_l", "Arm_L" }) },
        { "left_elbow", MakeAliases({ u8"\u5DE6\u3072\u3058", "left elbow", "LeftElbow", "elbow_l", "Elbow_L" }) },
        { "left_wrist", MakeAliases({ u8"\u5DE6\u624B\u9996", "left wrist", "LeftWrist", "wrist_l", "Wrist_L", u8"\u5DE6\u624B" }) },
        { "left_arm_twist", MakeAliases({ u8"\u5DE6\u8155\u6369", "left arm twist", "LeftArmTwist" }) },
        { "left_wrist_twist", MakeAliases({ u8"\u5DE6\u624B\u6369", "left wrist twist", "LeftWristTwist" }) },
        { "left_thumb0", MakeAliases({ u8"\u5DE6\u89AA\u6307\uFF10", u8"\u5DE6\u89AA\u63070", "left thumb0", "LeftThumb0" }) },
        { "left_thumb1", MakeAliases({ u8"\u5DE6\u89AA\u6307\uFF11", u8"\u5DE6\u89AA\u63071", "left thumb1", "LeftThumb1" }) },
        { "left_thumb2", MakeAliases({ u8"\u5DE6\u89AA\u6307\uFF12", u8"\u5DE6\u89AA\u63072", "left thumb2", "LeftThumb2" }) },
        { "left_index1", MakeAliases({ u8"\u5DE6\u4EBA\u6307\uFF11", u8"\u5DE6\u4EBA\u63071", "left index1", "LeftIndex1" }) },
        { "left_index2", MakeAliases({ u8"\u5DE6\u4EBA\u6307\uFF12", u8"\u5DE6\u4EBA\u63072", "left index2", "LeftIndex2" }) },
        { "left_index3", MakeAliases({ u8"\u5DE6\u4EBA\u6307\uFF13", u8"\u5DE6\u4EBA\u63073", "left index3", "LeftIndex3" }) },
        { "left_middle1", MakeAliases({ u8"\u5DE6\u4E2D\u6307\uFF11", u8"\u5DE6\u4E2D\u63071", "left middle1", "LeftMiddle1" }) },
        { "left_middle2", MakeAliases({ u8"\u5DE6\u4E2D\u6307\uFF12", u8"\u5DE6\u4E2D\u63072", "left middle2", "LeftMiddle2" }) },
        { "left_middle3", MakeAliases({ u8"\u5DE6\u4E2D\u6307\uFF13", u8"\u5DE6\u4E2D\u63073", "left middle3", "LeftMiddle3" }) },
        { "left_ring1", MakeAliases({ u8"\u5DE6\u85AC\u6307\uFF11", u8"\u5DE6\u85AC\u63071", "left ring1", "LeftRing1" }) },
        { "left_ring2", MakeAliases({ u8"\u5DE6\u85AC\u6307\uFF12", u8"\u5DE6\u85AC\u63072", "left ring2", "LeftRing2" }) },
        { "left_ring3", MakeAliases({ u8"\u5DE6\u85AC\u6307\uFF13", u8"\u5DE6\u85AC\u63073", "left ring3", "LeftRing3" }) },
        { "left_pinky1", MakeAliases({ u8"\u5DE6\u5C0F\u6307\uFF11", u8"\u5DE6\u5C0F\u63071", "left pinky1", "LeftPinky1" }) },
        { "left_pinky2", MakeAliases({ u8"\u5DE6\u5C0F\u6307\uFF12", u8"\u5DE6\u5C0F\u63072", "left pinky2", "LeftPinky2" }) },
        { "left_pinky3", MakeAliases({ u8"\u5DE6\u5C0F\u6307\uFF13", u8"\u5DE6\u5C0F\u63073", "left pinky3", "LeftPinky3" }) },
        { "right_shoulder_p", MakeAliases({ u8"\u53F3\u80A9P", "right shoulder p", "RightShoulderP" }) },
        { "right_shoulder", MakeAliases({ u8"\u53F3\u80A9", "right shoulder", "RightShoulder", "shoulder_r", "Shoulder_R" }) },
        { "right_arm", MakeAliases({ u8"\u53F3\u8155", "right arm", "RightArm", "arm_r", "Arm_R" }) },
        { "right_elbow", MakeAliases({ u8"\u53F3\u3072\u3058", "right elbow", "RightElbow", "elbow_r", "Elbow_R" }) },
        { "right_wrist", MakeAliases({ u8"\u53F3\u624B\u9996", "right wrist", "RightWrist", "wrist_r", "Wrist_R", u8"\u53F3\u624B" }) },
        { "right_arm_twist", MakeAliases({ u8"\u53F3\u8155\u6369", "right arm twist", "RightArmTwist" }) },
        { "right_wrist_twist", MakeAliases({ u8"\u53F3\u624B\u6369", "right wrist twist", "RightWristTwist" }) },
        { "right_thumb0", MakeAliases({ u8"\u53F3\u89AA\u6307\uFF10", u8"\u53F3\u89AA\u63070", "right thumb0", "RightThumb0" }) },
        { "right_thumb1", MakeAliases({ u8"\u53F3\u89AA\u6307\uFF11", u8"\u53F3\u89AA\u63071", "right thumb1", "RightThumb1" }) },
        { "right_thumb2", MakeAliases({ u8"\u53F3\u89AA\u6307\uFF12", u8"\u53F3\u89AA\u63072", "right thumb2", "RightThumb2" }) },
        { "right_index1", MakeAliases({ u8"\u53F3\u4EBA\u6307\uFF11", u8"\u53F3\u4EBA\u63071", "right index1", "RightIndex1" }) },
        { "right_index2", MakeAliases({ u8"\u53F3\u4EBA\u6307\uFF12", u8"\u53F3\u4EBA\u63072", "right index2", "RightIndex2" }) },
        { "right_index3", MakeAliases({ u8"\u53F3\u4EBA\u6307\uFF13", u8"\u53F3\u4EBA\u63073", "right index3", "RightIndex3" }) },
        { "right_middle1", MakeAliases({ u8"\u53F3\u4E2D\u6307\uFF11", u8"\u53F3\u4E2D\u63071", "right middle1", "RightMiddle1" }) },
        { "right_middle2", MakeAliases({ u8"\u53F3\u4E2D\u6307\uFF12", u8"\u53F3\u4E2D\u63072", "right middle2", "RightMiddle2" }) },
        { "right_middle3", MakeAliases({ u8"\u53F3\u4E2D\u6307\uFF13", u8"\u53F3\u4E2D\u63073", "right middle3", "RightMiddle3" }) },
        { "right_ring1", MakeAliases({ u8"\u53F3\u85AC\u6307\uFF11", u8"\u53F3\u85AC\u63071", "right ring1", "RightRing1" }) },
        { "right_ring2", MakeAliases({ u8"\u53F3\u85AC\u6307\uFF12", u8"\u53F3\u85AC\u63072", "right ring2", "RightRing2" }) },
        { "right_ring3", MakeAliases({ u8"\u53F3\u85AC\u6307\uFF13", u8"\u53F3\u85AC\u63073", "right ring3", "RightRing3" }) },
        { "right_pinky1", MakeAliases({ u8"\u53F3\u5C0F\u6307\uFF11", u8"\u53F3\u5C0F\u63071", "right pinky1", "RightPinky1" }) },
        { "right_pinky2", MakeAliases({ u8"\u53F3\u5C0F\u6307\uFF12", u8"\u53F3\u5C0F\u63072", "right pinky2", "RightPinky2" }) },
        { "right_pinky3", MakeAliases({ u8"\u53F3\u5C0F\u6307\uFF13", u8"\u53F3\u5C0F\u63073", "right pinky3", "RightPinky3" }) },
    };
}

saba::MMDNode* FindBoneNode(saba::MMDNodeManager* nodeMan, const std::vector<std::string>& aliases)
{
    for (const auto& alias : aliases)
    {
        auto* node = nodeMan->GetMMDNode(alias);
        if (node != nullptr)
        {
            return node;
        }
    }
    return nullptr;
}

void AppendMat4ColumnMajor(std::vector<float>* outValues, const glm::mat4& m)
{
    for (int col = 0; col < 4; col++)
    {
        for (int row = 0; row < 4; row++)
        {
            outValues->push_back(m[col][row]);
        }
    }
}

BonePosition FindBonePosition(saba::MMDNodeManager* nodeMan, const std::vector<std::string>& aliases)
{
    for (const auto& alias : aliases)
    {
        auto* node = nodeMan->GetMMDNode(alias);
        if (node != nullptr)
        {
            BonePosition result;
            result.found = true;
            result.value = ExtractTranslation(node->GetGlobalTransform());
            return result;
        }
    }
    return {};
}

void AppendBoneIfFound(std::vector<glm::vec3>* outPoints, const BonePosition& bone)
{
    if (bone.found)
    {
        outPoints->push_back(bone.value);
    }
}

RegionBounds BoundsFromPoints(const std::vector<glm::vec3>& points)
{
    if (points.empty())
    {
        return {};
    }

    glm::vec3 minPoint(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 maxPoint(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const auto& point : points)
    {
        minPoint = glm::min(minPoint, point);
        maxPoint = glm::max(maxPoint, point);
    }

    RegionBounds bounds;
    bounds.found = true;
    bounds.min = minPoint;
    bounds.max = maxPoint;
    return bounds;
}

RegionBounds ExpandBounds(const RegionBounds& bounds, const glm::vec3& padding)
{
    if (!bounds.found)
    {
        return bounds;
    }

    RegionBounds expanded = bounds;
    expanded.min -= padding;
    expanded.max += padding;
    return expanded;
}

RegionBounds ClampBoundsToFullBody(const RegionBounds& bounds, const glm::vec3& fullMin, const glm::vec3& fullMax)
{
    if (!bounds.found)
    {
        return bounds;
    }

    RegionBounds clamped = bounds;
    clamped.min = glm::max(clamped.min, fullMin);
    clamped.max = glm::min(clamped.max, fullMax);
    if (clamped.max.x < clamped.min.x || clamped.max.y < clamped.min.y || clamped.max.z < clamped.min.z)
    {
        return {};
    }
    return clamped;
}

json Vec3ToJson(const glm::vec3& v);

json BoundsToJson(const RegionBounds& bounds)
{
    return {
        { "min", Vec3ToJson(bounds.min) },
        { "max", Vec3ToJson(bounds.max) }
    };
}

RegionBounds MakeBoneDrivenRegionBounds(
    saba::MMDNodeManager* nodeMan,
    const glm::vec3& fullMin,
    const glm::vec3& fullMax,
    const char* regionName,
    bool legacyTargetsV1)
{
    const glm::vec3 fullSize = fullMax - fullMin;

    const auto hips = FindBonePosition(nodeMan, MakeAliases({
        u8"\u4E0B\u534A\u8EAB", "lower body", "LowerBody", u8"\u30BB\u30F3\u30BF\u30FC", "center", "Center", u8"\u30B0\u30EB\u30FC\u30D6", "groove"
    }));
    const auto upperBody = FindBonePosition(nodeMan, MakeAliases({
        u8"\u4E0A\u534A\u8EAB", "upper body", "UpperBody"
    }));
    const auto upperBody2 = FindBonePosition(nodeMan, MakeAliases({
        u8"\u4E0A\u534A\u8EAB2", "upper body2", "UpperBody2"
    }));
    const auto neck = FindBonePosition(nodeMan, MakeAliases({
        u8"\u9996", "neck", "Neck"
    }));
    const auto head = FindBonePosition(nodeMan, MakeAliases({
        u8"\u982D", "head", "Head"
    }));
    const auto leftShoulder = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u80A9", "left shoulder", "LeftShoulder", "shoulder_l", "Shoulder_L", u8"\u5DE6\u8155"
    }));
    const auto rightShoulder = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u80A9", "right shoulder", "RightShoulder", "shoulder_r", "Shoulder_R", u8"\u53F3\u8155"
    }));
    const auto leftElbow = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u3072\u3058", "left elbow", "LeftElbow", "elbow_l", "Elbow_L"
    }));
    const auto rightElbow = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u3072\u3058", "right elbow", "RightElbow", "elbow_r", "Elbow_R"
    }));
    const auto leftWrist = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u624B\u9996", "left wrist", "LeftWrist", "wrist_l", "Wrist_L", u8"\u5DE6\u624B"
    }));
    const auto rightWrist = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u624B\u9996", "right wrist", "RightWrist", "wrist_r", "Wrist_R", u8"\u53F3\u624B"
    }));
    const auto leftEye = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u76EE", "left eye", "LeftEye", "eye_l", "Eye_L"
    }));
    const auto rightEye = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u76EE", "right eye", "RightEye", "eye_r", "Eye_R"
    }));
    const auto leftLeg = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u8DB3", "left leg", "LeftLeg", "leg_l", "Leg_L"
    }));
    const auto rightLeg = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u8DB3", "right leg", "RightLeg", "leg_r", "Leg_R"
    }));
    const auto leftKnee = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u3072\u3056", "left knee", "LeftKnee", "knee_l", "Knee_L"
    }));
    const auto rightKnee = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u3072\u3056", "right knee", "RightKnee", "knee_r", "Knee_R"
    }));
    const auto leftAnkle = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u8DB3\u9996", "left ankle", "LeftAnkle", "ankle_l", "Ankle_L"
    }));
    const auto rightAnkle = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u8DB3\u9996", "right ankle", "RightAnkle", "ankle_r", "Ankle_R"
    }));
    const auto leftToe = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u3064\u307E\u5148", "left toe", "LeftToe", "toe_l", "Toe_L"
    }));
    const auto rightToe = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u3064\u307E\u5148", "right toe", "RightToe", "toe_r", "Toe_R"
    }));

    const float shoulderSpan = (leftShoulder.found && rightShoulder.found)
        ? glm::length(rightShoulder.value - leftShoulder.value)
        : fullSize.x * 0.42f;
    const float torsoHeight = (head.found && hips.found)
        ? glm::length(head.value - hips.value)
        : fullSize.y * 0.72f;

    std::vector<glm::vec3> points;
    RegionBounds region;

    const std::string regionKey(regionName);
    if (regionKey == "legs")
    {
        AppendBoneIfFound(&points, hips);
        AppendBoneIfFound(&points, leftLeg);
        AppendBoneIfFound(&points, rightLeg);
        AppendBoneIfFound(&points, leftKnee);
        AppendBoneIfFound(&points, rightKnee);
        AppendBoneIfFound(&points, leftAnkle);
        AppendBoneIfFound(&points, rightAnkle);
        AppendBoneIfFound(&points, leftToe);
        AppendBoneIfFound(&points, rightToe);
        region = BoundsFromPoints(points);
        region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.10f, torsoHeight * 0.05f, shoulderSpan * 0.10f));
    }
    else if (regionKey == "feet")
    {
        AppendBoneIfFound(&points, leftAnkle);
        AppendBoneIfFound(&points, rightAnkle);
        AppendBoneIfFound(&points, leftToe);
        AppendBoneIfFound(&points, rightToe);
        region = BoundsFromPoints(points);
        region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.12f, torsoHeight * 0.035f, shoulderSpan * 0.12f));
    }
    else if (regionKey == "knee_up")
    {
        AppendBoneIfFound(&points, leftKnee);
        AppendBoneIfFound(&points, rightKnee);
        AppendBoneIfFound(&points, hips);
        AppendBoneIfFound(&points, upperBody);
        AppendBoneIfFound(&points, upperBody2);
        AppendBoneIfFound(&points, neck);
        AppendBoneIfFound(&points, head);
        AppendBoneIfFound(&points, leftShoulder);
        AppendBoneIfFound(&points, rightShoulder);
        AppendBoneIfFound(&points, leftWrist);
        AppendBoneIfFound(&points, rightWrist);
        region = BoundsFromPoints(points);
        region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.12f, torsoHeight * 0.04f, shoulderSpan * 0.12f));
    }
    else if (regionKey == "abdomen")
    {
        const float y0 = fullMin.y + fullSize.y * 0.40f;
        const float y1 = fullMin.y + fullSize.y * 0.68f;
        const float centerX = (fullMin.x + fullMax.x) * 0.5f;
        const float centerZ = (fullMin.z + fullMax.z) * 0.5f;
        const float halfX = fullSize.x * 0.26f;
        const float halfZ = fullSize.z * 0.20f;
        region.found = true;
        region.min = glm::vec3(centerX - halfX, y0, centerZ - halfZ);
        region.max = glm::vec3(centerX + halfX, y1, centerZ + halfZ);
    }
    else if (regionKey == "waist_up")
    {
        AppendBoneIfFound(&points, hips);
        AppendBoneIfFound(&points, upperBody);
        AppendBoneIfFound(&points, upperBody2);
        AppendBoneIfFound(&points, neck);
        if (!legacyTargetsV1)
        {
            AppendBoneIfFound(&points, head);
        }
        AppendBoneIfFound(&points, leftShoulder);
        AppendBoneIfFound(&points, rightShoulder);
        AppendBoneIfFound(&points, leftElbow);
        AppendBoneIfFound(&points, rightElbow);
        region = BoundsFromPoints(points);
        if (legacyTargetsV1)
        {
            region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.12f, torsoHeight * 0.03f, shoulderSpan * 0.10f));
            return ClampBoundsToFullBody(region, fullMin, fullMax);
        }
        if (region.found && hips.found)
        {
            region.min.y = (std::min)(region.min.y, hips.value.y - torsoHeight * 0.16f);
        }
        if (region.found)
        {
            region.min.y = (std::min)(region.min.y, fullMin.y + fullSize.y * 0.45f);
        }
        region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.12f, torsoHeight * 0.04f, shoulderSpan * 0.10f));
    }
    else if (regionKey == "bust_up")
    {
        AppendBoneIfFound(&points, upperBody);
        AppendBoneIfFound(&points, upperBody2);
        AppendBoneIfFound(&points, neck);
        AppendBoneIfFound(&points, head);
        AppendBoneIfFound(&points, leftShoulder);
        AppendBoneIfFound(&points, rightShoulder);
        if (legacyTargetsV1)
        {
            AppendBoneIfFound(&points, leftElbow);
            AppendBoneIfFound(&points, rightElbow);
            region = BoundsFromPoints(points);
            region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.10f, torsoHeight * 0.03f, shoulderSpan * 0.08f));
            return ClampBoundsToFullBody(region, fullMin, fullMax);
        }
        region = BoundsFromPoints(points);
        if (region.found)
        {
            const BonePosition chest = upperBody2.found ? upperBody2 : upperBody;
            if (chest.found)
            {
                region.min.y = (std::min)(region.min.y, chest.value.y - torsoHeight * 0.05f);
            }
        }
        region = ExpandBounds(region, glm::vec3(shoulderSpan * 0.18f, torsoHeight * 0.04f, shoulderSpan * 0.12f));
        if (region.found)
        {
            const float centerX = (region.min.x + region.max.x) * 0.5f;
            const float centerZ = (region.min.z + region.max.z) * 0.5f;
            const float minHalfX = fullSize.x * 0.28f;
            const float minHalfZ = fullSize.z * 0.18f;
            region.min.x = (std::min)(region.min.x, centerX - minHalfX);
            region.max.x = (std::max)(region.max.x, centerX + minHalfX);
            region.min.z = (std::min)(region.min.z, centerZ - minHalfZ);
            region.max.z = (std::max)(region.max.z, centerZ + minHalfZ);

            const float shiftUp = torsoHeight * 0.04f;
            region.min.y += shiftUp * 0.5f;
            region.max.y += shiftUp;
        }
    }
    else if (regionKey == "face")
    {
        AppendBoneIfFound(&points, head);
        AppendBoneIfFound(&points, neck);
        AppendBoneIfFound(&points, leftEye);
        AppendBoneIfFound(&points, rightEye);
        region = BoundsFromPoints(points);
        if (region.found)
        {
            const glm::vec3 center = (region.min + region.max) * 0.5f;
            if (legacyTargetsV1)
            {
                const float faceWidth = (leftEye.found && rightEye.found)
                    ? glm::length(rightEye.value - leftEye.value) * 2.4f
                    : shoulderSpan * 0.34f;
                const float faceHeight = (head.found && neck.found)
                    ? glm::length(head.value - neck.value) * 1.2f
                    : torsoHeight * 0.18f;
                const float faceDepth = faceWidth * 0.75f;
                region.min = center - glm::vec3(faceWidth * 0.5f, faceHeight * 0.55f, faceDepth * 0.5f);
                region.max = center + glm::vec3(faceWidth * 0.5f, faceHeight * 0.45f, faceDepth * 0.5f);
                return ClampBoundsToFullBody(region, fullMin, fullMax);
            }
            const float faceWidth = (leftEye.found && rightEye.found)
                ? glm::length(rightEye.value - leftEye.value) * 2.65f
                : shoulderSpan * 0.38f;
            const float faceHeight = (head.found && neck.found)
                ? glm::length(head.value - neck.value) * 1.65f
                : torsoHeight * 0.24f;
            const float faceDepth = faceWidth * 0.85f;
            const glm::vec3 biasedCenter = center + glm::vec3(0.0f, faceHeight * 0.20f, 0.0f);
            region.min = biasedCenter - glm::vec3(faceWidth * 0.5f, faceHeight * 0.45f, faceDepth * 0.5f);
            region.max = biasedCenter + glm::vec3(faceWidth * 0.5f, faceHeight * 0.75f, faceDepth * 0.5f);
        }
    }

    return ClampBoundsToFullBody(region, fullMin, fullMax);
}

json Vec3ToJson(const glm::vec3& v)
{
    return json::array({ v.x, v.y, v.z });
}

glm::vec3 ComputeForwardFromBones(
    saba::MMDNodeManager* nodeMan,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax)
{
    const auto leftShoulder = FindBonePosition(nodeMan, MakeAliases({
        u8"\u5DE6\u80A9", "left shoulder", "LeftShoulder", "shoulder_l", "Shoulder_L", u8"\u5DE6\u8155"
    }));
    const auto rightShoulder = FindBonePosition(nodeMan, MakeAliases({
        u8"\u53F3\u80A9", "right shoulder", "RightShoulder", "shoulder_r", "Shoulder_R", u8"\u53F3\u8155"
    }));
    const auto head = FindBonePosition(nodeMan, MakeAliases({
        u8"\u982D", "head", "Head", u8"\u9996", "neck", "Neck"
    }));
    const auto hips = FindBonePosition(nodeMan, MakeAliases({
        u8"\u4E0B\u534A\u8EAB", "lower body", "LowerBody", u8"\u30BB\u30F3\u30BF\u30FC", "center", "Center", u8"\u30B0\u30EB\u30FC\u30D6", "groove"
    }));
    const auto upperBody = FindBonePosition(nodeMan, MakeAliases({
        u8"\u4E0A\u534A\u8EAB", "upper body", "UpperBody", u8"\u4E0A\u534A\u8EAB2", "UpperBody2"
    }));

    glm::vec3 up(0.0f, 1.0f, 0.0f);
    if (head.found && hips.found)
    {
        glm::vec3 delta = head.value - hips.value;
        if (glm::length(delta) > 1e-6f)
        {
            up = glm::normalize(delta);
        }
    }
    else if (head.found && upperBody.found)
    {
        glm::vec3 delta = head.value - upperBody.value;
        if (glm::length(delta) > 1e-6f)
        {
            up = glm::normalize(delta);
        }
    }

    if (leftShoulder.found && rightShoulder.found)
    {
        glm::vec3 right = rightShoulder.value - leftShoulder.value;
        right.y = 0.0f;
        if (glm::length(right) > 1e-6f)
        {
            right = glm::normalize(right);
            // MMD data and the downstream camera convention line up if actor forward
            // is derived with the opposite handedness from the naive shoulder cross.
            glm::vec3 forward = glm::cross(up, right);
            forward.y = 0.0f;
            if (glm::length(forward) > 1e-6f)
            {
                return glm::normalize(forward);
            }
        }
    }

    auto* upperNode = nodeMan->GetMMDNode(u8"\u4E0A\u534A\u8EAB");
    if (upperNode == nullptr)
    {
        upperNode = nodeMan->GetMMDNode("upper body");
    }
    if (upperNode == nullptr)
    {
        upperNode = nodeMan->GetMMDNode("UpperBody");
    }
    if (upperNode != nullptr)
    {
        glm::vec3 forward = -glm::vec3(upperNode->GetGlobalTransform()[2]);
        forward.y = 0.0f;
        if (glm::length(forward) > 1e-6f)
        {
            return glm::normalize(forward);
        }
    }

    (void)bboxMin;
    (void)bboxMax;
    return glm::vec3(0.0f, 0.0f, 1.0f);
}

bool DumpTargets(const Options& options)
{
    auto mmdModel = LoadModel(options.modelPath);
    if (mmdModel == nullptr)
    {
        std::cout << "Failed to load model.\n";
        return false;
    }

    std::unique_ptr<saba::VMDAnimation> vmdAnim;
    if (!LoadAnimation(mmdModel, options.vmdPaths, &vmdAnim))
    {
        std::cout << "Failed to load animation.\n";
        return false;
    }

    const int startFrame = (std::max)(0, options.startFrame);
    const int endFrame = options.endFrame >= 0 ? options.endFrame : int(vmdAnim->GetMaxKeyTime());
    if (endFrame < startFrame)
    {
        std::cout << "Invalid frame range.\n";
        return false;
    }

    auto updateFrame = [&](int frame) {
        mmdModel->BeginAnimation();
        mmdModel->UpdateAllAnimation(vmdAnim.get(), (float)frame, 1.0f / 30.0f);
        mmdModel->EndAnimation();
        mmdModel->Update();
    };

    mmdModel->InitializeAnimation();
    vmdAnim->SyncPhysics((float)startFrame);
    updateFrame(startFrame);

    json root;
    root["metadata"] = {
        { "model", options.modelPath },
        { "vmd", options.vmdPaths },
        { "start_frame", startFrame },
        { "end_frame", endFrame },
        { "step", options.step },
        { "fps", 30 }
    };
    root["frames"] = json::array();

    json globalRoot;
    std::vector<SimplifiedBoneSpec> simplifiedBoneSpecs;
    std::vector<saba::MMDNode*> simplifiedBoneNodes;
    std::vector<std::string> missingSimplifiedBones;
    const bool writeGlobalTransform = !options.globalTransformPath.empty();
    if (writeGlobalTransform)
    {
        simplifiedBoneSpecs = MakeSimplifiedBoneSpecs();
        auto* nodeMan = mmdModel->GetNodeManager();
        for (const auto& spec : simplifiedBoneSpecs)
        {
            auto* node = FindBoneNode(nodeMan, spec.aliases);
            simplifiedBoneNodes.push_back(node);
            if (node == nullptr)
            {
                missingSimplifiedBones.push_back(spec.boneName);
            }
        }

        globalRoot["BoneKeyFrameNumber"] = 0;
        globalRoot["BoneKeyFrameTransformRecord"] = json::array();
        globalRoot["BoneList"] = json::array();
        for (const auto& spec : simplifiedBoneSpecs)
        {
            globalRoot["BoneList"].push_back(spec.boneName);
        }
        globalRoot["metadata"] = {
            { "model", options.modelPath },
            { "vmd", options.vmdPaths },
            { "start_frame", startFrame },
            { "end_frame", endFrame },
            { "step", options.step },
            { "fps", 30 },
            { "format", "DanceCamAnimator-compatible simplified global transforms" },
            { "missing_bones", missingSimplifiedBones }
        };
    }

    bool hasPreviousForward = false;
    glm::vec3 previousForward(0.0f, 0.0f, 1.0f);
    bool hasPreviousReliableForward = false;
    glm::vec3 previousReliableForward(0.0f, 0.0f, 1.0f);
    const float flatPoseForwardRatioThreshold = 0.45f;

    auto appendFrame = [&](int frame) {
        const glm::vec3* positions = mmdModel->GetUpdatePositions();
        const size_t vertexCount = mmdModel->GetVertexCount();
        glm::vec3 bboxMin(FLT_MAX, FLT_MAX, FLT_MAX);
        glm::vec3 bboxMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (size_t i = 0; i < vertexCount; i++)
        {
            bboxMin = glm::min(bboxMin, positions[i]);
            bboxMax = glm::max(bboxMax, positions[i]);
        }

        glm::vec3 forward = ComputeForwardFromBones(mmdModel->GetNodeManager(), bboxMin, bboxMax);
        const glm::vec3 fullSize = bboxMax - bboxMin;
        const float horizontalSpan = (std::max)(fullSize.x, (std::max)(fullSize.z, 1e-6f));
        const float flatPoseRatio = fullSize.y / horizontalSpan;
        const bool flatPose = !options.legacyTargetsV1
            && flatPoseRatio < flatPoseForwardRatioThreshold
            && hasPreviousReliableForward;
        if (flatPose)
        {
            forward = previousReliableForward;
        }
        else
        {
            // Bone-derived forward is sign-ambiguous when the body lies nearly flat.
            // Keep adjacent frames continuous so "front" does not flip by ~180 deg.
            if (!options.legacyTargetsV1 && hasPreviousForward && glm::dot(forward, previousForward) < 0.0f)
            {
                forward = -forward;
            }
            if (!options.legacyTargetsV1 && flatPoseRatio >= flatPoseForwardRatioThreshold)
            {
                previousReliableForward = forward;
                hasPreviousReliableForward = true;
            }
        }
        previousForward = forward;
        hasPreviousForward = true;
        const RegionBounds legsBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "legs", options.legacyTargetsV1);
        const RegionBounds feetBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "feet", options.legacyTargetsV1);
        const RegionBounds kneeUpBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "knee_up", options.legacyTargetsV1);
        const RegionBounds abdomenBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "abdomen", options.legacyTargetsV1);
        const RegionBounds waistUpBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "waist_up", options.legacyTargetsV1);
        const RegionBounds bustUpBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "bust_up", options.legacyTargetsV1);
        const RegionBounds faceBounds = MakeBoneDrivenRegionBounds(mmdModel->GetNodeManager(), bboxMin, bboxMax, "face", options.legacyTargetsV1);

        json boundsJson = {
            {
                "full_body",
                {
                    { "min", Vec3ToJson(bboxMin) },
                    { "max", Vec3ToJson(bboxMax) }
                }
            }
        };
        if (legsBounds.found)
        {
            boundsJson["legs"] = BoundsToJson(legsBounds);
        }
        if (!options.legacyTargetsV1 && feetBounds.found)
        {
            boundsJson["feet"] = BoundsToJson(feetBounds);
        }
        if (kneeUpBounds.found)
        {
            boundsJson["knee_up"] = BoundsToJson(kneeUpBounds);
        }
        if (!options.legacyTargetsV1 && abdomenBounds.found)
        {
            boundsJson["abdomen"] = BoundsToJson(abdomenBounds);
        }
        if (waistUpBounds.found)
        {
            boundsJson["waist_up"] = BoundsToJson(waistUpBounds);
        }
        if (bustUpBounds.found)
        {
            boundsJson["bust_up"] = BoundsToJson(bustUpBounds);
        }
        if (faceBounds.found)
        {
            boundsJson["face"] = BoundsToJson(faceBounds);
        }

        root["frames"].push_back({
            { "frame", frame },
            { "forward", Vec3ToJson(forward) },
            { "bounds", boundsJson }
        });

        if (writeGlobalTransform)
        {
            std::vector<float> transforms;
            transforms.reserve(simplifiedBoneNodes.size() * 16);
            for (auto* node : simplifiedBoneNodes)
            {
                if (node != nullptr)
                {
                    AppendMat4ColumnMajor(&transforms, node->GetGlobalTransform());
                }
                else
                {
                    AppendMat4ColumnMajor(&transforms, glm::mat4(1.0f));
                }
            }
            globalRoot["BoneKeyFrameTransformRecord"].push_back({
                { "FrameTime", frame },
                { "Transform", transforms }
            });
        }
    };

    appendFrame(startFrame);

    for (int frame = startFrame + 1; frame <= endFrame; frame++)
    {
        updateFrame(frame);

        if (((frame - startFrame) % options.step) == 0)
        {
            appendFrame(frame);
        }
    }

    if (!options.outputPath.empty())
    {
        std::ofstream out(options.outputPath, std::ios::binary);
        if (!out.is_open())
        {
            std::cout << "Failed to open output JSON.\n";
            return false;
        }
        out << root.dump(2);
        out.close();

        std::cout
            << "Wrote target data: " << options.outputPath
            << " frames=" << root["frames"].size()
            << " range=" << startFrame << "-" << endFrame
            << " step=" << options.step << "\n";
    }

    if (writeGlobalTransform)
    {
        globalRoot["BoneKeyFrameNumber"] = globalRoot["BoneKeyFrameTransformRecord"].size();
        std::ofstream globalOut(options.globalTransformPath, std::ios::binary);
        if (!globalOut.is_open())
        {
            std::cout << "Failed to open global transform JSON.\n";
            return false;
        }
        globalOut << globalRoot.dump(2);
        globalOut.close();
        std::cout
            << "Wrote global transform data: " << options.globalTransformPath
            << " frames=" << globalRoot["BoneKeyFrameNumber"]
            << " missing_bones=" << missingSimplifiedBones.size()
            << "\n";
    }
    return true;
}
}

int main(int argc, char** argv)
{
    std::vector<std::string> args(argc);
#if _WIN32
    {
        WCHAR* cmdline = GetCommandLineW();
        int wArgc;
        WCHAR** wArgs = CommandLineToArgvW(cmdline, &wArgc);
        if (wArgs == nullptr)
        {
            std::cout << "Failed to parse command line.\n";
            return 1;
        }
        for (int i = 0; i < argc && i < wArgc; i++)
        {
            args[i] = saba::ToUtf8String(wArgs[i]);
        }
        LocalFree(wArgs);
    }
#else
    for (int i = 0; i < argc; i++)
    {
        args[i] = argv[i];
    }
#endif

    Options options;
    if (!ParseOptions(args, &options))
    {
        Usage();
        return 1;
    }

    if (!DumpTargets(options))
    {
        return 1;
    }

    return 0;
}
