/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx].x += dL_dmean.x;
	dL_dmeans[idx].y += dL_dmean.y;
	dL_dmeans[idx].z += dL_dmean.z;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot
	)
{

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	glm::vec3 dL_dmean2;
	float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
	dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
	dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_dmean2;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ alphas,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dpixel_median_depths,
	const float* __restrict__ dL_dpixel_depth_vars,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	float2* dgndcs_dview,
	float* __restrict__ dL_dview,
	float* dg_camd_dviewmatrix,
	const float3* means,
	const float* view,
	glm::vec3* dL_dmeans,
	const float* __restrict__ gt_depth,
	bool track_off,
	bool map_off
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float accum_depth_rec = 0;
	float dL_dpixel_depth;
	float dL_dpixel_median_depth;
	float dL_dpixel_depth_var;
	float gt_px_depth;
	float accum_alpha_rec = 0;
	float accum_var_rec = 0;
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dpixel_median_depth = dL_dpixel_median_depths[pix_id];
		dL_dpixel_depth_var = dL_dpixel_depth_vars[pix_id];
		gt_px_depth = gt_depth[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	float last_var = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	float dL_dv0 = 0;
	float dL_dv1 = 0;
	float dL_dv2 = 0;
	float dL_dv4 = 0;
	float dL_dv5 = 0;
	float dL_dv6 = 0;
	float dL_dv8 = 0;
	float dL_dv9 = 0;
	float dL_dv10 = 0;
	float dL_dv12 = 0;
	float dL_dv13 = 0;
	float dL_dv14 = 0;
	bool mid_once = true;
	int max_id = -1;
	float max_terminate_p = 0.0f;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 15.0f / 255.0f)
				continue;
			
			// if (depths[collected_id[j]] < 0.9 * gt_px_depth || depths[collected_id[j]] > 1.1 * gt_px_depth)
			// {
			// 	continue;
			// }

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;
			const float dL_ddepth = dpixel_depth_ddepth * dL_dpixel_depth;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			// float dL_dalpha_color = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// dL_dalpha_color += (c - accum_rec[ch]) * dL_dchannel;


				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				if (!map_off)
				{
					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				}
				// atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			
			// Propagate gradients from pixel depth to opacity
			const float c_d = collected_depths[j];
			const float c_var = (c_d - gt_px_depth) * (c_d - gt_px_depth);
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			accum_var_rec = last_alpha * last_var + (1.f - last_alpha) * accum_var_rec;
			last_var = c_var;

			dL_dalpha += (c_d - accum_depth_rec) * dL_dpixel_depth;
			dL_dalpha += (c_var - accum_var_rec) * dL_dpixel_depth_var;
			if(!map_off)
			{
				atomicAdd(&(dL_ddepths[global_id]), dL_ddepth + dL_dpixel_depth_var * dpixel_depth_ddepth * 2. * (c_d - gt_px_depth));
			}
			// dL_dalpha += (c_var - accum_var_rec) * dL_dpixel_depth_var;
			// atomicAdd(&(dL_ddepths[global_id]), dL_ddepth + dL_dpixel_depth_var * dpixel_depth_ddepth * 2. * (c_d - gt_px_depth));

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			if (!track_off)
			{
				float dL_dndcs_x = dL_dalpha * con_o.w * dG_ddelx * ddelx_dx;
				float dL_dndcs_y = dL_dalpha * con_o.w * dG_ddely * ddely_dy;

				//TODO: add pose gradient w.r.t. depth rendering
				dL_dv0 += dgndcs_dview[global_id * 12 + 0].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 0].y * dL_dndcs_y;
				dL_dv1 += dgndcs_dview[global_id * 12 + 1].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 1].y * dL_dndcs_y;
				dL_dv2 += dgndcs_dview[global_id * 12 + 2].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 2].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 0] * dL_ddepth;
				dL_dv4 += dgndcs_dview[global_id * 12 + 3].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 3].y * dL_dndcs_y;
				dL_dv5 += dgndcs_dview[global_id * 12 + 4].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 4].y * dL_dndcs_y;
				dL_dv6 += dgndcs_dview[global_id * 12 + 5].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 5].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 1] * dL_ddepth;
				dL_dv8 += dgndcs_dview[global_id * 12 + 6].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 6].y * dL_dndcs_y;
				dL_dv9 += dgndcs_dview[global_id * 12 + 7].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 7].y * dL_dndcs_y;
				dL_dv10 += dgndcs_dview[global_id * 12 + 8].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 8].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 2] * dL_ddepth;
				dL_dv12 += dgndcs_dview[global_id * 12 + 9].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 9].y * dL_dndcs_y;
				dL_dv13 += dgndcs_dview[global_id * 12 + 10].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 10].y * dL_dndcs_y;
				dL_dv14 += dgndcs_dview[global_id * 12 + 11].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 11].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 3] * dL_ddepth;
			}
			

			if(!map_off)
			{
				if (T > 0.5f && mid_once)
				{	
					float mul3 = view[2] * means[global_id].x + view[6] * means[global_id].y + view[10] * means[global_id].z + view[14];
					atomicAdd(&(dL_dmeans[global_id].x), (view[2] - view[3] * mul3) * dL_dpixel_median_depth * 1.0f);
					atomicAdd(&(dL_dmeans[global_id].y), (view[6] - view[7] * mul3) * dL_dpixel_median_depth * 1.0f);
					atomicAdd(&(dL_dmeans[global_id].z), (view[10] - view[11] * mul3) * dL_dpixel_median_depth * 1.0f);
					mid_once = false;
				}
			}
			// if (T > 0.5f && mid_once)
			// {	
			// 	float mul3 = view[2] * means[global_id].x + view[6] * means[global_id].y + view[10] * means[global_id].z + view[14];
			// 	atomicAdd(&(dL_dmeans[global_id].x), (view[2] - view[3] * mul3) * dL_dpixel_median_depth * 1.0f);
			// 	atomicAdd(&(dL_dmeans[global_id].y), (view[6] - view[7] * mul3) * dL_dpixel_median_depth * 1.0f);
			// 	atomicAdd(&(dL_dmeans[global_id].z), (view[10] - view[11] * mul3) * dL_dpixel_median_depth * 1.0f);
			// 	mid_once = false;
			// }
			// if (alpha * T > max_terminate_p)
			// {	
			// 	max_id = global_id;
			// 	max_terminate_p = alpha * T;
			// }



			// atomicAdd(&(dL_dview[0]), dgndcs_dview[global_id * 12 + 0].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 0].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[1]), dgndcs_dview[global_id * 12 + 1].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 1].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[2]), dgndcs_dview[global_id * 12 + 2].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 2].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 0] * dL_ddepth);
			// atomicAdd(&(dL_dview[4]), dgndcs_dview[global_id * 12 + 3].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 3].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[5]), dgndcs_dview[global_id * 12 + 4].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 4].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[6]), dgndcs_dview[global_id * 12 + 5].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 5].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 1] * dL_ddepth);
			// atomicAdd(&(dL_dview[8]), dgndcs_dview[global_id * 12 + 6].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 6].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[9]), dgndcs_dview[global_id * 12 + 7].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 7].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[10]), dgndcs_dview[global_id * 12 + 8].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 8].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 2] * dL_ddepth);
			// atomicAdd(&(dL_dview[12]), dgndcs_dview[global_id * 12 + 9].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 9].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[13]), dgndcs_dview[global_id * 12 + 10].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 10].y * dL_dndcs_y);
			// atomicAdd(&(dL_dview[14]), dgndcs_dview[global_id * 12 + 11].x * dL_dndcs_x + dgndcs_dview[global_id * 12 + 11].y * dL_dndcs_y + dg_camd_dviewmatrix[global_id * 4 + 3] * dL_ddepth);
			
			if(!map_off)
			{
				// Update gradients w.r.t. 2D mean position of the Gaussian
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

				// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
				atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
				atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
				atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

				// Update gradients w.r.t. opacity of the Gaussian
				atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
				
			}

			// // Update gradients w.r.t. 2D mean position of the Gaussian
			// atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			// atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// // Update gradients w.r.t. opacity of the Gaussian
			// atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
	// if (max_id > 0)
	// {
	// 	float mul3 = view[2] * means[max_id].x + view[6] * means[max_id].y + view[10] * means[max_id].z + view[14];
	// 	atomicAdd(&(dL_dmeans[max_id].x), (view[2] - view[3] * mul3) * dL_dpixel_median_depth * 1.0f);
	// 	atomicAdd(&(dL_dmeans[max_id].y), (view[6] - view[7] * mul3) * dL_dpixel_median_depth * 1.0f);
	// 	atomicAdd(&(dL_dmeans[max_id].z), (view[10] - view[11] * mul3) * dL_dpixel_median_depth * 1.0f);
	// 	// atomicAdd(&(dL_dmeans[max_id].x), (view[2]) * dL_dpixel_median_depth * 1.0f);
	// 	// atomicAdd(&(dL_dmeans[max_id].y), (view[6]) * dL_dpixel_median_depth * 1.0f);
	// 	// atomicAdd(&(dL_dmeans[max_id].z), (view[10]) * dL_dpixel_median_depth * 1.0f);
	// }
	
	// if (!track_off)
	// {
	// 	atomicAdd(&(dL_dview[0]), dL_dv0);
	// 	atomicAdd(&(dL_dview[1]), dL_dv1);
	// 	atomicAdd(&(dL_dview[2]), dL_dv2);
	// 	atomicAdd(&(dL_dview[4]), dL_dv4);
	// 	atomicAdd(&(dL_dview[5]), dL_dv5);
	// 	atomicAdd(&(dL_dview[6]), dL_dv6);
	// 	atomicAdd(&(dL_dview[8]), dL_dv8);
	// 	atomicAdd(&(dL_dview[9]), dL_dv9);
	// 	atomicAdd(&(dL_dview[10]), dL_dv10);
	// 	atomicAdd(&(dL_dview[12]), dL_dv12);
	// 	atomicAdd(&(dL_dview[13]), dL_dv13);
	// 	atomicAdd(&(dL_dview[14]), dL_dv14);
	// }
	if ((!track_off) && inside)
	{
		atomicAdd(&(dL_dview[pix_id*16 + 0]), dL_dv0);
		atomicAdd(&(dL_dview[pix_id*16 + 1]), dL_dv1);
		atomicAdd(&(dL_dview[pix_id*16 + 2]), dL_dv2);
		atomicAdd(&(dL_dview[pix_id*16 + 4]), dL_dv4);
		atomicAdd(&(dL_dview[pix_id*16 + 5]), dL_dv5);
		atomicAdd(&(dL_dview[pix_id*16 + 6]), dL_dv6);
		atomicAdd(&(dL_dview[pix_id*16 + 8]), dL_dv8);
		atomicAdd(&(dL_dview[pix_id*16 + 9]), dL_dv9);
		atomicAdd(&(dL_dview[pix_id*16 + 10]), dL_dv10);
		atomicAdd(&(dL_dview[pix_id*16 + 12]), dL_dv12);
		atomicAdd(&(dL_dview[pix_id*16 + 13]), dL_dv13);
		atomicAdd(&(dL_dview[pix_id*16 + 14]), dL_dv14);
	}
	
}

template<int C>
__global__ void pose_gradient_preCUDA(
	int P,
	const float3* means,
	const int* radii,
	const float* view,
	const float* proj,
	float2* dgndcs_dview,
	const float* perspec,
	float* dg_camd_dviewmatrix
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// sort: v0:0 /v1:1 /v2:2 /v4:3 /v5:4 /v6:5 /v8:6 /v9:7 /v10:8 /v12:9 /v13:10 /v14:11
	// x_ndc w.r.t v0,v1,v2,v4,v5,v6,v8,v9,v10,v12,v13,v14
	dgndcs_dview[idx*12 + 0].x = m_w * perspec[0] * m.x; 
	dgndcs_dview[idx*12 + 3].x = m_w * perspec[0] * m.y;
	dgndcs_dview[idx*12 + 6].x = m_w * perspec[0] * m.z;
	dgndcs_dview[idx*12 + 9].x = m_w * perspec[0] * 1.0f;

	dgndcs_dview[idx*12 + 2].x = m_hom.x * (-m_w * m_w) * m.x;
	dgndcs_dview[idx*12 + 5].x = m_hom.x * (-m_w * m_w) * m.y;
	dgndcs_dview[idx*12 + 8].x = m_hom.x * (-m_w * m_w) * m.z;
	dgndcs_dview[idx*12 + 11].x = m_hom.x * (-m_w * m_w) * 1.0f;

	// y_ndc w.r.t v0,v1,v2,v4,v5,v6,v8,v9,v10,v12,v13,v14
	dgndcs_dview[idx*12 + 1].y = m_w * perspec[5] * m.x;
	dgndcs_dview[idx*12 + 4].y = m_w * perspec[5] * m.y;
	dgndcs_dview[idx*12 + 7].y = m_w * perspec[5] * m.z; 
	dgndcs_dview[idx*12 + 10].y = m_w * perspec[5] * 1.0f; 

	dgndcs_dview[idx*12 + 2].y = m_hom.y * (-m_w * m_w) * m.x;
	dgndcs_dview[idx*12 + 5].y = m_hom.y * (-m_w * m_w) * m.y;
	dgndcs_dview[idx*12 + 8].y = m_hom.y * (-m_w * m_w) * m.z;
	dgndcs_dview[idx*12 + 11].y = m_hom.y * (-m_w * m_w) * 1.0f;

	dg_camd_dviewmatrix[idx*4 + 0] = m.x;   //v2
	dg_camd_dviewmatrix[idx*4 + 1] = m.y;   //v6
	dg_camd_dviewmatrix[idx*4 + 2] = m.z;   //v10
	dg_camd_dviewmatrix[idx*4 + 3] = 1.0f;   //v14
	
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot
	)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot
		);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* alphas,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixel_depths,
	const float* dL_dpixel_median_depths,
	const float* dL_dpixel_depth_vars,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	float* dgc_ndcs_dviewmatrix,
	float* dL_dview,
	float* dg_camd_dviewmatrix,
	const float3* means3D,
	const float* viewmatrix,
	glm::vec3* dL_dmean3D,
	const float* gt_depth,
	bool track_off,
	bool map_off
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		alphas,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dpixel_median_depths,
		dL_dpixel_depth_vars,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths,
		(float2*)dgc_ndcs_dviewmatrix,
		dL_dview,
		dg_camd_dviewmatrix,
		(float3*)means3D,
		viewmatrix,
		dL_dmean3D,
		gt_depth,
		track_off,
		map_off
		);
}

void BACKWARD::pose_gradient_pre(
	int P,
	const float3* means3D,
	const int* radii,
	const float* viewmatrix,
	const float* projmatrix,
	float* dgc_ndcs_dviewmatrix,
	const float* perspec_matrix,
	float* dg_camd_dviewmatrix
	)
{
	pose_gradient_preCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, 
		(float3*)means3D,
		radii,
		viewmatrix,
		projmatrix,
		(float2*)dgc_ndcs_dviewmatrix,
		perspec_matrix,
		dg_camd_dviewmatrix
	);
}