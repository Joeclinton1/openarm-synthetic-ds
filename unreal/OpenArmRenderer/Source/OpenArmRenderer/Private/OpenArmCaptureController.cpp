#include "OpenArmCaptureController.h"

#include "Dom/JsonObject.h"
#include "Engine/Scene.h"
#include "EngineUtils.h"
#include "HAL/PlatformMisc.h"
#include "HAL/IConsoleManager.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "MuJoCo/Components/Sensors/MjCamera.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

AOpenArmCaptureController::AOpenArmCaptureController()
{
    PrimaryActorTick.bCanEverTick = true;
}

void AOpenArmCaptureController::BeginPlay()
{
    Super::BeginPlay();
    bLoaded = LoadJob();
}

void AOpenArmCaptureController::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (bLoaded && !bConfigured)
    {
        bConfigured = ConfigureCameras();
        if (bConfigured)
        {
            SetActorTickEnabled(false);
        }
    }
}

bool AOpenArmCaptureController::LoadJob()
{
    FString JobPath;
    FParse::Value(FCommandLine::Get(), TEXT("OpenArmJob="), JobPath);
    if (JobPath.IsEmpty())
    {
        JobPath = FPlatformMisc::GetEnvironmentVariable(TEXT("OPENARM_URLAB_JOB"));
    }
    if (JobPath.IsEmpty())
    {
        UE_LOG(LogTemp, Error, TEXT("OpenArm renderer requires -OpenArmJob=/absolute/urlab_job.json"));
        return false;
    }

    FString Text;
    if (!FFileHelper::LoadFileToString(Text, *JobPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Cannot read OpenArm job: %s"), *JobPath);
        return false;
    }
    TSharedPtr<FJsonObject> Root;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid()
        || Root->GetStringField(TEXT("schema")) != TEXT("openarm-urlab-job-v2"))
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid OpenArm URLab v2 job: %s"), *JobPath);
        return false;
    }

    const TSharedPtr<FJsonObject> Camera = Root->GetObjectField(TEXT("camera"));
    const TArray<TSharedPtr<FJsonValue>>& Resolution = Camera->GetArrayField(TEXT("resolution"));
    const TArray<TSharedPtr<FJsonValue>>& Rows = Camera->GetArrayField(TEXT("intrinsics"));
    Width = Resolution[0]->AsNumber();
    Height = Resolution[1]->AsNumber();
    const TArray<TSharedPtr<FJsonValue>>& Row0 = Rows[0]->AsArray();
    const TArray<TSharedPtr<FJsonValue>>& Row1 = Rows[1]->AsArray();
    Fx = Row0[0]->AsNumber();
    Skew = Row0[1]->AsNumber();
    Cx = Row0[2]->AsNumber();
    Fy = Row1[1]->AsNumber();
    Cy = Row1[2]->AsNumber();
    for (const auto& Pair : Camera->GetObjectField(TEXT("streams"))->Values)
    {
        Streams.Add(Pair.Key, Pair.Value->AsString());
    }
    const TSharedPtr<FJsonObject> Render = Root->GetObjectField(TEXT("render"));
    bHardwareRayTracing = Render->GetBoolField(TEXT("hardware_ray_tracing"));
    LumenQuality = Render->GetObjectField(TEXT("lumen"))->GetIntegerField(TEXT("quality"));
    if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(
            TEXT("r.Lumen.ScreenProbeGather.Quality")))
    {
        CVar->Set(LumenQuality, ECVF_SetByCommandline);
    }
    if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(
            TEXT("r.Lumen.HardwareRayTracing")))
    {
        CVar->Set(bHardwareRayTracing ? 1 : 0, ECVF_SetByCommandline);
    }
    return Width > 0 && Height > 0 && Fx > 0.0 && Fy > 0.0;
}

FMatrix AOpenArmCaptureController::MakeProjectionMatrix() const
{
    // Infinite reversed-Z perspective in Unreal's row-vector convention.
    // Principal point and skew come directly from the OpenCV K matrix.
    FMatrix P = FMatrix::Identity;
    FMemory::Memzero(P.M, sizeof(P.M));
    P.M[0][0] = 2.0 * Fx / Width;
    P.M[1][0] = 2.0 * Skew / Width;
    P.M[1][1] = 2.0 * Fy / Height;
    P.M[2][0] = 1.0 - 2.0 * Cx / Width;
    P.M[2][1] = 2.0 * Cy / Height - 1.0;
    P.M[2][3] = 1.0;
    P.M[3][2] = 0.1; // one millimetre in Unreal centimetres
    return P;
}

void AOpenArmCaptureController::ConfigureCamera(UMjCamera* Camera, const FString& StreamName)
{
    const bool bWasStreaming = Camera->IsStreamingActive();
    if (bWasStreaming)
    {
        Camera->SetStreamingEnabled(false);
    }
    Camera->resolution = {Width, Height};
    if (StreamName == TEXT("depth_m"))
    {
        Camera->CaptureMode = EMjCameraMode::Depth;
    }
    else if (StreamName == TEXT("instance_segmentation"))
    {
        Camera->CaptureMode = EMjCameraMode::InstanceSegmentation;
    }
    else
    {
        Camera->CaptureMode = EMjCameraMode::Real;
    }

    if (Camera->CaptureComponent)
    {
        USceneCaptureComponent2D* Capture = Camera->CaptureComponent;
        Capture->bUseCustomProjectionMatrix = true;
        Capture->CustomProjectionMatrix = MakeProjectionMatrix();
        Capture->bUseRayTracingIfEnabled = bHardwareRayTracing;
        Capture->ShowFlags.SetMotionBlur(false);
        Capture->ShowFlags.SetDepthOfField(false);
        Capture->PostProcessSettings.bOverride_AutoExposureMethod = true;
        Capture->PostProcessSettings.AutoExposureMethod = EAutoExposureMethod::AEM_Manual;
        Capture->PostProcessSettings.bOverride_AutoExposureBias = true;
        Capture->PostProcessSettings.AutoExposureBias = 0.0f;
        Capture->PostProcessSettings.bOverride_MotionBlurAmount = true;
        Capture->PostProcessSettings.MotionBlurAmount = 0.0f;
    }
    if (bWasStreaming)
    {
        Camera->SetStreamingEnabled(true);
    }
}

bool AOpenArmCaptureController::ConfigureCameras()
{
    int32 Configured = 0;
    for (TObjectIterator<UMjCamera> It; It; ++It)
    {
        UMjCamera* Camera = *It;
        if (!Camera || Camera->GetWorld() != GetWorld())
        {
            continue;
        }
        for (const auto& Pair : Streams)
        {
            if (Camera->MjName == Pair.Value || Camera->GetName().Contains(Pair.Value))
            {
                ConfigureCamera(Camera, Pair.Key);
                ++Configured;
            }
        }
    }
    if (Configured == Streams.Num())
    {
        UE_LOG(LogTemp, Display, TEXT("Configured %d synchronized OpenArm capture streams"), Configured);
        return true;
    }
    return false; // Imported articulation may not have completed BeginPlay yet.
}
