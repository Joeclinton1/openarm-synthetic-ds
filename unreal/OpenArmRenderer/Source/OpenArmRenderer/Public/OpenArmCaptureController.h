#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "OpenArmCaptureController.generated.h"

class UMjCamera;

/**
 * Configures the imported URLab cameras from an openarm-urlab-job-v2 file.
 * URLab owns synchronous readback and transport; this actor owns the exact
 * off-centre projection and fixed production post-process contract.
 */
UCLASS()
class OPENARMRENDERER_API AOpenArmCaptureController : public AActor
{
    GENERATED_BODY()

public:
    AOpenArmCaptureController();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;

private:
    bool ConfigureCameras();
    bool LoadJob();
    void ConfigureCamera(UMjCamera* Camera, const FString& StreamName);
    FMatrix MakeProjectionMatrix() const;

    bool bLoaded = false;
    bool bConfigured = false;
    int32 Width = 0;
    int32 Height = 0;
    double Fx = 0.0;
    double Fy = 0.0;
    double Skew = 0.0;
    double Cx = 0.0;
    double Cy = 0.0;
    bool bHardwareRayTracing = false;
    int32 LumenQuality = 2;
    TMap<FString, FString> Streams;
};
