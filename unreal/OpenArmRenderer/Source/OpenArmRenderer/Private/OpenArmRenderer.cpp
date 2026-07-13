#include "Modules/ModuleManager.h"
#include "OpenArmCaptureController.h"

class FOpenArmRendererModule final : public FDefaultGameModuleImpl
{
public:
    virtual void StartupModule() override
    {
        WorldHandle = FWorldDelegates::OnPostWorldInitialization.AddLambda(
            [](UWorld* World, const UWorld::InitializationValues) {
                if (World && World->IsGameWorld())
                {
                    World->SpawnActor<AOpenArmCaptureController>();
                }
            });
    }

    virtual void ShutdownModule() override
    {
        FWorldDelegates::OnPostWorldInitialization.Remove(WorldHandle);
    }

private:
    FDelegateHandle WorldHandle;
};

IMPLEMENT_PRIMARY_GAME_MODULE(FOpenArmRendererModule, OpenArmRenderer, "OpenArmRenderer");
