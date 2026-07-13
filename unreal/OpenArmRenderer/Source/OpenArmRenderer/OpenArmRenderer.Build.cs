using UnrealBuildTool;

public class OpenArmRenderer : ModuleRules
{
    public OpenArmRenderer(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] {
            "Core", "CoreUObject", "Engine", "Json", "JsonUtilities", "RenderCore", "URLab"
        });
    }
}
