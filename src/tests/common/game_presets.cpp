// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <memory>
#include <string>
#include <catch2/catch_test_macros.hpp>
#include "common/game_presets.h"

namespace {
constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;

struct Oot900pTestState {
    Settings::Values& settings{Settings::values};
    const Settings::ResolutionSetup previous_resolution{settings.resolution_setup.GetValue()};
    const Settings::RendererBackend previous_backend{settings.renderer_backend.GetValue()};
    const Settings::ConsoleMode previous_console_mode{settings.use_docked_mode.GetValue()};
    const Settings::ResolutionScalingInfo previous_resolution_info{settings.resolution_info};
    const u64 previous_program_id{Settings::GetCurrentProgramID()};
    std::string previous_environment;
    bool had_environment{};

    Oot900pTestState() {
        if (const char* const value = std::getenv("EDEN_OOT_900P")) {
            previous_environment = value;
            had_environment = true;
        }
    }

    static void SetEnvironment(const char* value) {
#ifdef _WIN32
        _putenv_s("EDEN_OOT_900P", value ? value : "");
#else
        if (value) {
            setenv("EDEN_OOT_900P", value, 1);
        } else {
            unsetenv("EDEN_OOT_900P");
        }
#endif
    }

    ~Oot900pTestState() {
        SetEnvironment(had_environment ? previous_environment.c_str() : nullptr);
        Settings::SetCurrentProgramID(previous_program_id);
        settings.resolution_setup.SetValue(previous_resolution);
        settings.renderer_backend.SetValue(previous_backend);
        settings.use_docked_mode.SetValue(previous_console_mode);
        settings.resolution_info = previous_resolution_info;
    }

    void Configure(bool enabled, u64 program_id = OcarinaOfTimeProgramId,
                   Settings::ConsoleMode console_mode = Settings::ConsoleMode::Docked,
                   Settings::RendererBackend backend = Settings::RendererBackend::Vulkan,
                   Settings::ResolutionSetup setup = Settings::ResolutionSetup::Res1X) {
        SetEnvironment(enabled ? "1" : "0");
        Settings::SetCurrentProgramID(program_id);
        settings.use_docked_mode.SetValue(console_mode);
        settings.renderer_backend.SetValue(backend);
        settings.resolution_setup.SetValue(setup);
    }
};

void SetConflictingPreferences(Settings::Values& settings, bool per_game) {
    const auto set = [per_game](auto& setting, auto value) {
        setting.SetGlobal(!per_game);
        setting.SetValue(value);
    };
    set(settings.renderer_backend, Settings::RendererBackend::Null);
    set(settings.resolution_setup, Settings::ResolutionSetup::Res2X);
    set(settings.gpu_accuracy, Settings::GpuAccuracy::High);
    set(settings.use_asynchronous_shaders, false);
    set(settings.astc_recompression, Settings::AstcRecompression::Uncompressed);
    set(settings.vram_usage_mode, Settings::VramUsageMode::Aggressive);
    set(settings.use_reactive_flushing, true);
    set(settings.vsync_mode, Settings::VSyncMode::Immediate);
    set(settings.scaling_filter, Settings::ScalingFilter::NearestNeighbor);
    set(settings.anti_aliasing, Settings::AntiAliasing::Smaa);
}

void CheckPreset(const Settings::Values& settings) {
    CHECK(settings.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);
    CHECK(settings.resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
    CHECK(settings.gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK(settings.use_asynchronous_shaders.GetValue());
    CHECK(settings.astc_recompression.GetValue() == Settings::AstcRecompression::Bc3);
    CHECK(settings.vram_usage_mode.GetValue() == Settings::VramUsageMode::Conservative);
    CHECK_FALSE(settings.use_reactive_flushing.GetValue());
    CHECK(settings.vsync_mode.GetValue() == Settings::VSyncMode::FifoRelaxed);
    CHECK(settings.use_docked_mode.GetValue() == Settings::ConsoleMode::Handheld);
    CHECK(settings.scaling_filter.GetValue() == Settings::ScalingFilter::ScaleForce);
    CHECK(settings.anti_aliasing.GetValue() == Settings::AntiAliasing::Fxaa);
}

void CheckPreferences(const Settings::Values& settings, bool global = false) {
    CHECK(settings.renderer_backend.GetValue(global) == Settings::RendererBackend::Null);
    CHECK(settings.resolution_setup.GetValue(global) == Settings::ResolutionSetup::Res2X);
    CHECK(settings.gpu_accuracy.GetValue(global) == Settings::GpuAccuracy::High);
    CHECK_FALSE(settings.use_asynchronous_shaders.GetValue(global));
    CHECK(settings.astc_recompression.GetValue(global) ==
          Settings::AstcRecompression::Uncompressed);
    CHECK(settings.vram_usage_mode.GetValue(global) == Settings::VramUsageMode::Aggressive);
    CHECK(settings.use_reactive_flushing.GetValue(global));
    CHECK(settings.vsync_mode.GetValue(global) == Settings::VSyncMode::Immediate);
    CHECK(settings.scaling_filter.GetValue(global) == Settings::ScalingFilter::NearestNeighbor);
    CHECK(settings.anti_aliasing.GetValue(global) == Settings::AntiAliasing::Smaa);
}
} // namespace

TEST_CASE("Display filter defaults use ScaleForce and FXAA", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    CHECK(settings->scaling_filter.GetDefault() == Settings::ScalingFilter::ScaleForce);
    CHECK(settings->anti_aliasing.GetDefault() == Settings::AntiAliasing::Fxaa);
}

TEST_CASE("Accepted MKW settings replace visible global preferences", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    Settings::SetMarioKartWorldRecommendedSettings(*settings, true);
    CheckPreset(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->gpu_accuracy.UsingGlobal());
    CHECK(settings->renderer_backend.GetValue(true) == Settings::RendererBackend::Vulkan);
    CHECK(settings->gpu_accuracy.GetValue(true) == Settings::GpuAccuracy::Low);
}

TEST_CASE("MKW preset overrides saved settings for the session", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    SetConflictingPreferences(*settings, true);
    REQUIRE(
        Settings::ApplyMarioKartWorldPreset(*settings, Settings::MarioKartWorldProgramId, true));
    CheckPreset(*settings);
    CheckPreferences(*settings, true);
    for (const auto& restore : settings->linkage.restore_functions) {
        restore();
    }
    CheckPreferences(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->resolution_setup.UsingGlobal());
    REQUIRE(
        Settings::ApplyMarioKartWorldPreset(*settings, Settings::MarioKartWorldProgramId, true));
    CheckPreset(*settings);
}

TEST_CASE("MKW preset does not change unrelated games", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    CHECK_FALSE(Settings::ApplyMarioKartWorldPreset(*settings, 0, true));
    CHECK_FALSE(Settings::ApplyMarioKartWorldPreset(*settings,
                                                    Settings::MarioKartWorldProgramId + 1, true));
    CheckPreferences(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->resolution_setup.UsingGlobal());
}

TEST_CASE("Declining the MKW preset preserves global and per-game preferences", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    for (bool per_game : {false, true}) {
        SetConflictingPreferences(*settings, per_game);
        CHECK_FALSE(Settings::ApplyMarioKartWorldPreset(*settings,
                                                        Settings::MarioKartWorldProgramId, false));
        CheckPreferences(*settings);
        CHECK(settings->renderer_backend.UsingGlobal() == !per_game);
        CHECK(settings->resolution_setup.UsingGlobal() == !per_game);
    }
}

TEST_CASE("OOT 900p override uses exact 5/8 scaling only for its supported setup",
          "[game_presets][resolution]") {
    SECTION("applies 5/8 for docked Vulkan OOT at 1X") {
        Oot900pTestState state;
        state.Configure(true);

        Settings::UpdateRescalingInfo();

        const auto& info = Settings::values.resolution_info;
        CHECK(Settings::values.resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
        CHECK(info.up_scale == 5);
        CHECK(info.down_shift == 3);
        CHECK(info.up_factor == 0.625f);
        CHECK(info.down_factor == 1.6f);
        CHECK(info.active);
        CHECK(info.downscale);
        CHECK(info.ScaleUp(2560U) == 1600);
        CHECK(info.ScaleUp(1440U) == 900);
        CHECK(info.ScaleUp(0) == 0);

        Oot900pTestState::SetEnvironment("0");
        Settings::UpdateRescalingInfo();
        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("ignores an invalid environment value") {
        Oot900pTestState state;
        state.Configure(false);
        Oot900pTestState::SetEnvironment("1 ");

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("an absent environment variable leaves the base scale unchanged") {
        Oot900pTestState state;
        state.Configure(false);
        Oot900pTestState::SetEnvironment(nullptr);

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("the exact value 0 disables the override") {
        Oot900pTestState state;
        state.Configure(false);

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("does not apply to another title") {
        Oot900pTestState state;
        state.Configure(true, OcarinaOfTimeProgramId + 1);

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("does not apply in handheld mode") {
        Oot900pTestState state;
        state.Configure(true, OcarinaOfTimeProgramId, Settings::ConsoleMode::Handheld);

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("does not apply to a non-Vulkan backend") {
        Oot900pTestState state;
        state.Configure(true, OcarinaOfTimeProgramId, Settings::ConsoleMode::Docked,
                        Settings::RendererBackend::Null);

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 1);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK_FALSE(Settings::values.resolution_info.active);
    }

    SECTION("does not replace an explicit non-1X setup") {
        Oot900pTestState state;
        state.Configure(true, OcarinaOfTimeProgramId, Settings::ConsoleMode::Docked,
                        Settings::RendererBackend::Vulkan, Settings::ResolutionSetup::Res2X);

        Settings::UpdateRescalingInfo();

        CHECK(Settings::values.resolution_info.up_scale == 2);
        CHECK(Settings::values.resolution_info.down_shift == 0);
        CHECK(Settings::values.resolution_info.up_factor == 2.0f);
        CHECK(Settings::values.resolution_info.down_factor == 0.5f);
        CHECK(Settings::values.resolution_info.active);
        CHECK_FALSE(Settings::values.resolution_info.downscale);
    }
}
