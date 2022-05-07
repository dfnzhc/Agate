//
// Created by 秋鱼 on 2022/4/30.
//

#pragma once

#include <memory>
#include "Camera.hpp"

namespace Agate {

enum class ViewMode
{
    EyeFixed, Orbit
};

inline float radians(float degrees)
{
    return degrees * M_PIf / 180.0f;
}
inline float degrees(float radians)
{
    return radians * M_1_PIf * 180.0f;
}

class MouseTracker
{
    Camera* camera_ = nullptr;
    ViewMode view_mode_ = ViewMode::EyeFixed;
    float lookat_distance_ = 0.0f;
    float move_speed_ = 1.0f;
    const float ZoomFactor = 1.1f;

    // in radians
    float pitch_ = 0.0f;
    float yaw_ = 0.0f;

    int prev_posX_ = 0;
    int prev_posY_ = 0;
    bool start_tracking_ = false;

    float3 u_ = {0.0f, 0.0f, 0.0f};
    float3 v_ = {0.0f, 0.0f, 0.0f};
    float3 w_ = {0.0f, 0.0f, 0.0f};

public:
    void setViewMode(ViewMode viewMode) { view_mode_ = viewMode; }

    void setCamera(Camera* camera)
    {
        camera_ = camera;
        initFromCamera();
    }
    const Camera* getCamera() const { return camera_; }

    float moveSpeed() { return move_speed_; }
    void setMoveSpeed(float newSpeed) { move_speed_ = newSpeed; }

    void zoom(int dir)
    {
        if (dir == 0)
            return;

        float zoomSize = dir > 0 ? 1.0f / ZoomFactor : ZoomFactor;
        lookat_distance_ *= zoomSize;
        const float3& lookat = camera_->lookat;
        const float3& eye = camera_->eye;

        camera_->eye = lookat + (eye - lookat) * zoomSize;
    }

    void startTracking(int x, int y)
    {
        prev_posX_ = x;
        prev_posY_ = y;
        start_tracking_ = true;
    }

    void stopTracking()
    {
        start_tracking_ = false;
    }

    void update(int x, int y)
    {
        if (!start_tracking_) {
            startTracking(x, y);
            return;
        }

        auto deltaX = static_cast<float>(x - prev_posX_);
        auto deltaY = static_cast<float>(y - prev_posY_);
        prev_posX_ = x;
        prev_posY_ = y;
        pitch_ = radians(std::min(89.0f, std::max(-89.0f, degrees(pitch_) + 0.5f * deltaY)));
        yaw_ = radians(fmod(degrees(yaw_) - 0.5f * deltaX, 360.0f));

        updateCamera();
    }

    void setFrame(const float3& u, const float3& v, const float3& w)
    {
        u_ = u;
        v_ = v;
        w_ = w;

        float3 viewDirWorld = -normalize(camera_->lookat - camera_->eye);
        float3 viewDirLocal;
        viewDirLocal.x = dot(viewDirWorld, u);
        viewDirLocal.y = dot(viewDirWorld, v);
        viewDirLocal.z = dot(viewDirWorld, w);

        pitch_ = asin(viewDirLocal.z);
        yaw_ = atan2(viewDirLocal.x, viewDirLocal.y);
    }

private:
    void initFromCamera()
    {
        camera_->UVWFrame(u_, v_, w_);
        normalize(u_);
        normalize(v_);
        normalize(w_);

        std::swap(v_, w_);
        pitch_ = 0.0f;
        yaw_ = 0.0f;
        lookat_distance_ = length(camera_->lookat - camera_->eye);
    }

    void updateCamera()
    {
        float3 viewDirLocal;
        viewDirLocal.x = cos(pitch_) * sin(yaw_);
        viewDirLocal.y = cos(pitch_) * cos(yaw_);
        viewDirLocal.z = sin(pitch_);

        float3 viewDirWorld = u_ * viewDirLocal.x + v_ * viewDirLocal.y + w_ * viewDirLocal.z;

        if (view_mode_ == ViewMode::EyeFixed) {
            const float3& eye = camera_->eye;
            camera_->lookat = eye - viewDirWorld * lookat_distance_;
        } else if (view_mode_ == ViewMode::Orbit) {
            const float3& lookat = camera_->lookat;
            camera_->eye = lookat + viewDirWorld * lookat_distance_;
        }
    }
};

} // namespace Agate
