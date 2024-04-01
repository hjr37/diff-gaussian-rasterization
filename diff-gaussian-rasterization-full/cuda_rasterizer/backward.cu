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
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs, glm::vec3* dgc_dCampos)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	float len_dir_orig3 = glm::length(dir_orig) * glm::length(dir_orig) * glm::length(dir_orig);
	float one_len_dir_orig3 = 1.0f / len_dir_orig3;
	float one_len_dir_orig = 1.0f / glm::length(dir_orig);

	float dxdCamx = dir_orig.x * dir_orig.x * one_len_dir_orig3 - one_len_dir_orig;
	float dydCamx = dir_orig.x * dir_orig.y * one_len_dir_orig3;
	float dzdCamx = dir_orig.x * dir_orig.z * one_len_dir_orig3;
	
	float dxdCamy = dir_orig.x * dir_orig.y * one_len_dir_orig3;
	float dydCamy = dir_orig.y * dir_orig.y * one_len_dir_orig3 - one_len_dir_orig;
	float dzdCamy = dir_orig.y * dir_orig.z * one_len_dir_orig3;

	float dxdCamz = dir_orig.x * dir_orig.z * one_len_dir_orig3;
	float dydCamz = dir_orig.y * dir_orig.z * one_len_dir_orig3;
	float dzdCamz = dir_orig.z * dir_orig.z * one_len_dir_orig3 - one_len_dir_orig;
	

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	// float dL_dgau_single_depth = dL_dgau_depths[idx];

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

	glm::vec3 dRGBdCamx = dRGBdx * dxdCamx + dRGBdy * dydCamx + dRGBdz * dzdCamx;
	glm::vec3 dRGBdCamy = dRGBdx * dxdCamy + dRGBdy * dydCamy + dRGBdz * dzdCamy;
	glm::vec3 dRGBdCamz = dRGBdx * dxdCamz + dRGBdy * dydCamz + dRGBdz * dzdCamz;

	glm::vec3* single_dgc_dCampos = dgc_dCampos + idx * 3;
	single_dgc_dCampos[0] = dRGBdCamx;
	single_dgc_dCampos[1] = dRGBdCamy;
	single_dgc_dCampos[2] = dRGBdCamz;

	//end
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
	float* dL_dcov, 
	float3* dginvcovs_dT,
	float* dL_dgau_depths
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.

	float dL_dgau_single_depth = dL_dgau_depths[idx];

	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1; // bool flag if gaussian out of frustrum
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
		float dcon_o_x_da = denom2inv * (-c * c);
		float dcon_o_x_db = denom2inv * (2 * b * c);
		float dcon_o_x_dc = denom2inv * (-b * b);

		float dcon_o_y_da = denom2inv * (b * c);
		float dcon_o_y_db = denom2inv * (-a * c - b * b);
		float dcon_o_y_dc = denom2inv * (a * b);

		float dcon_o_w_da = denom2inv * (-b * b);
		float dcon_o_w_db = denom2inv * (2 * a * b);
		float dcon_o_w_dc = denom2inv * (-a * a);

		float da_dT00 = 2 * cov3D[0] * T[0][0] + 2 * cov3D[1] * T[0][1] + 2 * cov3D[2] * T[0][2]; 
		float da_dT01 = 2 * cov3D[1] * T[0][0] + 2 * cov3D[3] * T[0][1] + 2 * cov3D[4] * T[0][2];
		float da_dT02 = 2 * cov3D[2] * T[0][0] + 2 * cov3D[4] * T[0][1] + 2 * cov3D[5] * T[0][2];
		float da_dT10 = 0;
		float da_dT11 = 0;
		float da_dT12 = 0;

		float db_dT00 = cov3D[0] * T[1][0] + cov3D[1] * T[1][1] + cov3D[2] * T[1][2];
		float db_dT01 = cov3D[1] * T[1][0] + cov3D[3] * T[1][1] + cov3D[4] * T[1][2];
		float db_dT02 = cov3D[2] * T[1][0] + cov3D[4] * T[1][1] + cov3D[5] * T[1][2];
		float db_dT10 = cov3D[0] * T[0][0] + cov3D[1] * T[0][1] + cov3D[2] * T[0][2];
		float db_dT11 = cov3D[1] * T[0][0] + cov3D[3] * T[0][1] + cov3D[4] * T[0][2];
		float db_dT12 = cov3D[2] * T[0][0] + cov3D[4] * T[0][1] + cov3D[5] * T[0][2];

		float dc_dT00 = 0;
		float dc_dT01 = 0;
		float dc_dT02 = 0;
		float dc_dT10 = 2 * cov3D[0] * T[1][0] + 2 * cov3D[1] * T[1][1] + 2 * cov3D[2] * T[1][2];
		float dc_dT11 = 2 * cov3D[1] * T[1][0] + 2 * cov3D[3] * T[1][1] + 2 * cov3D[4] * T[1][2];
		float dc_dT12 = 2 * cov3D[2] * T[1][0] + 2 * cov3D[4] * T[1][1] + 2 * cov3D[5] * T[1][2];

		float dcon_o_x_dT00 = dcon_o_x_da * da_dT00 + dcon_o_x_db * db_dT00 + dcon_o_x_dc * dc_dT00;
		float dcon_o_x_dT01 = dcon_o_x_da * da_dT01 + dcon_o_x_db * db_dT01 + dcon_o_x_dc * dc_dT01;
		float dcon_o_x_dT02 = dcon_o_x_da * da_dT02 + dcon_o_x_db * db_dT02 + dcon_o_x_dc * dc_dT02;
		float dcon_o_x_dT10 = dcon_o_x_da * da_dT10 + dcon_o_x_db * db_dT10 + dcon_o_x_dc * dc_dT10;
		float dcon_o_x_dT11 = dcon_o_x_da * da_dT11 + dcon_o_x_db * db_dT11 + dcon_o_x_dc * dc_dT11;
		float dcon_o_x_dT12 = dcon_o_x_da * da_dT12 + dcon_o_x_db * db_dT12 + dcon_o_x_dc * dc_dT12;

		float dcon_o_y_dT00 = dcon_o_y_da * da_dT00 + dcon_o_y_db * db_dT00 + dcon_o_y_dc * dc_dT00;
		float dcon_o_y_dT01 = dcon_o_y_da * da_dT01 + dcon_o_y_db * db_dT01 + dcon_o_y_dc * dc_dT01;
		float dcon_o_y_dT02 = dcon_o_y_da * da_dT02 + dcon_o_y_db * db_dT02 + dcon_o_y_dc * dc_dT02;
		float dcon_o_y_dT10 = dcon_o_y_da * da_dT10 + dcon_o_y_db * db_dT10 + dcon_o_y_dc * dc_dT10;
		float dcon_o_y_dT11 = dcon_o_y_da * da_dT11 + dcon_o_y_db * db_dT11 + dcon_o_y_dc * dc_dT11;
		float dcon_o_y_dT12 = dcon_o_y_da * da_dT12 + dcon_o_y_db * db_dT12 + dcon_o_y_dc * dc_dT12;

		float dcon_o_w_dT00 = dcon_o_w_da * da_dT00 + dcon_o_w_db * db_dT00 + dcon_o_w_dc * dc_dT00;
		float dcon_o_w_dT01 = dcon_o_w_da * da_dT01 + dcon_o_w_db * db_dT01 + dcon_o_w_dc * dc_dT01;
		float dcon_o_w_dT02 = dcon_o_w_da * da_dT02 + dcon_o_w_db * db_dT02 + dcon_o_w_dc * dc_dT02;
		float dcon_o_w_dT10 = dcon_o_w_da * da_dT10 + dcon_o_w_db * db_dT10 + dcon_o_w_dc * dc_dT10;
		float dcon_o_w_dT11 = dcon_o_w_da * da_dT11 + dcon_o_w_db * db_dT11 + dcon_o_w_dc * dc_dT11;
		float dcon_o_w_dT12 = dcon_o_w_da * da_dT12 + dcon_o_w_db * db_dT12 + dcon_o_w_dc * dc_dT12;

		dginvcovs_dT[idx * 6 +0] = {dcon_o_x_dT00, dcon_o_y_dT00, dcon_o_w_dT00};
		dginvcovs_dT[idx * 6 +1] = {dcon_o_x_dT01, dcon_o_y_dT01, dcon_o_w_dT01};
		dginvcovs_dT[idx * 6 +2] = {dcon_o_x_dT02, dcon_o_y_dT02, dcon_o_w_dT02};
		dginvcovs_dT[idx * 6 +3] = {dcon_o_x_dT10, dcon_o_y_dT10, dcon_o_w_dT10};
		dginvcovs_dT[idx * 6 +4] = {dcon_o_x_dT11, dcon_o_y_dT11, dcon_o_w_dT11};
		dginvcovs_dT[idx * 6 +5] = {dcon_o_x_dT12, dcon_o_y_dT12, dcon_o_w_dT12};
	}
	else
	{
		dginvcovs_dT[idx * 6 +0] = {0, 0, 0};
		dginvcovs_dT[idx * 6 +1] = {0, 0, 0};
		dginvcovs_dT[idx * 6 +2] = {0, 0, 0};
		dginvcovs_dT[idx * 6 +3] = {0, 0, 0};
		dginvcovs_dT[idx * 6 +4] = {0, 0, 0};
		dginvcovs_dT[idx * 6 +5] = {0, 0, 0};
	}
	

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
	dL_dmeans[idx] = dL_dmean;

	float mul3 = view_matrix[2] * mean.x + view_matrix[6] * mean.y + view_matrix[10] * mean.z + view_matrix[14];
	dL_dmeans[idx] = add_f3_f3(dL_dmeans[idx], float3{dL_dgau_single_depth * (view_matrix[2]-view_matrix[3] * mul3), dL_dgau_single_depth * (view_matrix[6]-view_matrix[7] * mul3), dL_dgau_single_depth * (view_matrix[10]-view_matrix[11] * mul3)});
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
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dgc_dCampos,
	const float* perspec, 
	float2* dgndcs_dview
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

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, (glm::vec3*)dgc_dCampos);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);

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

	dgndcs_dview[idx*12 + 1].y = m_w * perspec[5] * m.x;
	dgndcs_dview[idx*12 + 4].y = m_w * perspec[5] * m.y;
	dgndcs_dview[idx*12 + 7].y = m_w * perspec[5] * m.z; 
	dgndcs_dview[idx*12 + 10].y = m_w * perspec[5] * 1.0f; 

	dgndcs_dview[idx*12 + 2].y = m_hom.y * (-m_w * m_w) * m.x;
	dgndcs_dview[idx*12 + 5].y = m_hom.y * (-m_w * m_w) * m.y;
	dgndcs_dview[idx*12 + 8].y = m_hom.y * (-m_w * m_w) * m.z;
	dgndcs_dview[idx*12 + 11].y = m_hom.y * (-m_w * m_w) * 1.0f;
	
	//end
}

// Backward version.x of the rendering procedure.
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
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors, 
	float* __restrict__ dpix_dgc, 
	int* __restrict__ gau_id_l, 
	int* __restrict__ pix_id_l, 
	const uint32_t* __restrict__ n_valid_contrib_cumsum, 
	float3* __restrict__ dpixel_dndcs, 
	float3* __restrict__ dpixel_dinvcovs,
	float* __restrict__ dL_dgau_depths,
	float* __restrict__ ddepth_dndcs, 
	float* __restrict__ ddepth_dinvcovs,
	const float* __restrict__ gt_depth,
	const float* __restrict__ dL_duncertainties
	//end
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
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float dpixel_dalpha[C] = { 0 };
	float ddepth_dalpha = 0;
	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float dL_depth;
	float dL_duncertainty;
	float gt_px_depth;
	float accum_depth_rec = 0;
	float accum_uncertainty_rec = 0;
	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_depth = dL_depths[pix_id];
		dL_duncertainty = dL_duncertainties[pix_id];
		gt_px_depth = gt_depth[pix_id];
	}
	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	float last_uncertainty = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	int seq = 0;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1]; //start in the BACK
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

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

				dpixel_dalpha[ch] = T * (c - accum_rec[ch]);

				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			const float c_d = collected_depths[j];
			const float c_u = (collected_depths[j] - gt_px_depth) * (collected_depths[j] - gt_px_depth);
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			accum_uncertainty_rec = last_alpha * last_uncertainty + (1.f - last_alpha) * accum_uncertainty_rec;
			last_depth = c_d;
			last_uncertainty = c_u;
			dL_dalpha += (c_d - accum_depth_rec) * dL_depth;
			dL_dalpha += (c_u - accum_uncertainty_rec) * dL_duncertainty;
			atomicAdd(&(dL_dgau_depths[global_id]), dchannel_dcolor * dL_depth + 2. * (collected_depths[j] - gt_px_depth) * dchannel_dcolor * dL_duncertainty);
			ddepth_dalpha = T * (c_d - accum_depth_rec);

			if (pix_id == 0)
			{
			dpix_dgc[(0 + seq)] = dchannel_dcolor; 
			gau_id_l[(0 + seq)] = global_id;
			pix_id_l[(0 + seq)] = pix_id;
			}
			else
			{
			dpix_dgc[(n_valid_contrib_cumsum[pix_id-1] + seq)] = dchannel_dcolor;
			gau_id_l[(n_valid_contrib_cumsum[pix_id-1] + seq)] = global_id;
			pix_id_l[(n_valid_contrib_cumsum[pix_id-1] + seq)] = pix_id;
			}
			//end

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

			if (pix_id == 0)
			{
				dpixel_dndcs[(0 + seq)*2 + 0]= {
					(dpixel_dalpha[0] * con_o.w * dG_ddelx * ddelx_dx),
					(dpixel_dalpha[1] * con_o.w * dG_ddelx * ddelx_dx),
					(dpixel_dalpha[2] * con_o.w * dG_ddelx * ddelx_dx),
				};
		
				dpixel_dndcs[(0 + seq)*2 + 1] = {
					(dpixel_dalpha[0] * con_o.w * dG_ddely * ddely_dy),
					(dpixel_dalpha[1] * con_o.w * dG_ddely * ddely_dy),
					(dpixel_dalpha[2] * con_o.w * dG_ddely * ddely_dy),
				};

				ddepth_dndcs[(0 + seq)*2 + 0] = ddepth_dalpha * con_o.w * dG_ddelx * ddelx_dx;
				ddepth_dndcs[(0 + seq)*2 + 1] = ddepth_dalpha * con_o.w * dG_ddely * ddely_dy;
			}
			else
			{
				dpixel_dndcs[(n_valid_contrib_cumsum[pix_id-1] + seq)*2 + 0] = {
					(dpixel_dalpha[0] * con_o.w * dG_ddelx * ddelx_dx),
					(dpixel_dalpha[1] * con_o.w * dG_ddelx * ddelx_dx),
					(dpixel_dalpha[2] * con_o.w * dG_ddelx * ddelx_dx),
				};

				dpixel_dndcs[(n_valid_contrib_cumsum[pix_id-1] + seq)*2 + 1] = {
					(dpixel_dalpha[0] * con_o.w * dG_ddely * ddely_dy),
					(dpixel_dalpha[1] * con_o.w * dG_ddely * ddely_dy),
					(dpixel_dalpha[2] * con_o.w * dG_ddely * ddely_dy),
				};

				ddepth_dndcs[(n_valid_contrib_cumsum[pix_id-1] + seq)*2 + 0] = ddepth_dalpha * con_o.w * dG_ddelx * ddelx_dx;
				ddepth_dndcs[(n_valid_contrib_cumsum[pix_id-1] + seq)*2 + 1] = ddepth_dalpha * con_o.w * dG_ddely * ddely_dy;
			}

			float3 dpixel_dcon_o_x = {
				dpixel_dalpha[0] * con_o.w * gdx * d.x * (-0.5f),
				dpixel_dalpha[1] * con_o.w * gdx * d.x * (-0.5f),
				dpixel_dalpha[2] * con_o.w * gdx * d.x * (-0.5f),
			};

			float3 dpixel_dcon_o_w = {
				dpixel_dalpha[0] * con_o.w * gdy * d.y * (-0.5f),
				dpixel_dalpha[1] * con_o.w * gdy * d.y * (-0.5f),
				dpixel_dalpha[2] * con_o.w * gdy * d.y * (-0.5f),
			};

			float3 dpixel_dcon_o_y = {
				dpixel_dalpha[0] * con_o.w * gdx * d.y * (-1.0f),
				dpixel_dalpha[1] * con_o.w * gdx * d.y * (-1.0f),
				dpixel_dalpha[2] * con_o.w * gdx * d.y * (-1.0f),
			};
			
			if (pix_id == 0)
			{
				dpixel_dinvcovs[(0 + seq)*3 + 0] = dpixel_dcon_o_x;
				dpixel_dinvcovs[(0 + seq)*3 + 1] = dpixel_dcon_o_y;
				dpixel_dinvcovs[(0 + seq)*3 + 2] = dpixel_dcon_o_w;

				ddepth_dinvcovs[(0 + seq)*3 + 0] = ddepth_dalpha * con_o.w * gdx * d.x * (-0.5f);
				ddepth_dinvcovs[(0 + seq)*3 + 1] = ddepth_dalpha * con_o.w * gdx * d.y * (-1.0f);
				ddepth_dinvcovs[(0 + seq)*3 + 2] = ddepth_dalpha * con_o.w * gdy * d.y * (-0.5f);
			}
			else
			{
				dpixel_dinvcovs[(n_valid_contrib_cumsum[pix_id-1] + seq)*3 + 0] = dpixel_dcon_o_x;
				dpixel_dinvcovs[(n_valid_contrib_cumsum[pix_id-1] + seq)*3 + 1] = dpixel_dcon_o_y;
				dpixel_dinvcovs[(n_valid_contrib_cumsum[pix_id-1] + seq)*3 + 2] = dpixel_dcon_o_w;

				ddepth_dinvcovs[(n_valid_contrib_cumsum[pix_id-1] + seq)*3 + 0] = ddepth_dalpha * con_o.w * gdx * d.x * (-0.5f);
				ddepth_dinvcovs[(n_valid_contrib_cumsum[pix_id-1] + seq)*3 + 1] = ddepth_dalpha * con_o.w * gdx * d.y * (-1.0f);
				ddepth_dinvcovs[(n_valid_contrib_cumsum[pix_id-1] + seq)*3 + 2] = ddepth_dalpha * con_o.w * gdy * d.y * (-0.5f);
			}

			seq++;
			


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
	}
}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
ComputePG(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float3* __restrict__ means3D, 
	const float* __restrict__ viewmatrix,
	const float f_x, float f_y,
	float* __restrict__ dpix_dgc, 
	glm::vec3* __restrict__ dgc_dCampos,
	int* __restrict__ gau_id_list, 
	int* __restrict__ pix_id_list, 
	const uint32_t* __restrict__ n_valid_contrib_cumsum,
	const uint32_t* __restrict__ n_valid_contrib, 
	float3* __restrict__ dpixel_dndcs, 
	float2* __restrict__ dgndcs_dviewmatrix, 
	float3* __restrict__ dpixel_dinvcovs, 
	float3* __restrict__ dginvcovs_dT, 
	float* __restrict__ dL_dview,
	const float* __restrict__ dL_dpixels,
	float* __restrict__ ddepth_dndcs, 
	float* __restrict__ ddepth_dinvcovs,
	const float* __restrict__ dL_depths
	)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;

	// debug cuda memory illegal access
	if (!inside)
	{
		return;
	}
	// end 

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	uint32_t contributor = toDo;

	bool shut = false;
	int valid_seq = 0;
	int length;
	int* single_pix_gs_id;
	float3* single_dpixel_dndcs;
	float3* single_dpixel_dinvcovs;
	float* single_dpixel_dgc;

	float* single_ddepth_dndcs;
	float* single_ddepth_dinvcovs;
	
	if (pix_id == 0)
	{
		int offset = 0;
		length = n_valid_contrib[pix_id];
		single_pix_gs_id = gau_id_list + offset;

		// single_dpixel_dndcs = &dpixel_dndcs[offset * 2];
		// single_dpixel_dinvcovs = &dpixel_dinvcovs[offset * 3];
		// single_dpixel_dgc = &dpix_dgc[offset];
		single_dpixel_dndcs = dpixel_dndcs + offset * 2;
		single_dpixel_dinvcovs = dpixel_dinvcovs + offset * 3;
		single_dpixel_dgc = dpix_dgc + offset;

		single_ddepth_dndcs = ddepth_dndcs + offset * 2;
		single_ddepth_dinvcovs = ddepth_dinvcovs + offset * 3;
	}
	else
	{
		int offset = n_valid_contrib_cumsum[pix_id-1];
		length = n_valid_contrib[pix_id];
		single_pix_gs_id = gau_id_list + offset;
		
		// single_dpixel_dndcs = &dpixel_dndcs[offset * 2];
		// single_dpixel_dinvcovs = &dpixel_dinvcovs[offset * 3];
		// single_dpixel_dgc = &dpix_dgc[offset];
		single_dpixel_dndcs = dpixel_dndcs + offset * 2;
		single_dpixel_dinvcovs = dpixel_dinvcovs + offset * 3;
		single_dpixel_dgc = dpix_dgc + offset;

		single_ddepth_dndcs = ddepth_dndcs + offset * 2;
		single_ddepth_dinvcovs = ddepth_dinvcovs + offset * 3;
	}

	// cuda memory illegal access debug
	if (length == 0)
	{
		return;
	}

	float3 dp_dv0 = {0, 0, 0};
	float3 dp_dv1 = {0, 0, 0};
	float3 dp_dv2 = {0, 0, 0};
	float3 dp_dv4 = {0, 0, 0};
	float3 dp_dv5 = {0, 0, 0};
	float3 dp_dv6 = {0, 0, 0};
	float3 dp_dv8 = {0, 0, 0};
	float3 dp_dv9 = {0, 0, 0};
	float3 dp_dv10 = {0, 0, 0};
	float3 dp_dv12 = {0, 0, 0};
	float3 dp_dv13 = {0, 0, 0};
	float3 dp_dv14 = {0, 0, 0};

	float dd_dv0 = 0;
	float dd_dv1 = 0;
	float dd_dv2 = 0;
	float dd_dv4 = 0;
	float dd_dv5 = 0;
	float dd_dv6 = 0;
	float dd_dv8 = 0;
	float dd_dv9 = 0;
	float dd_dv10 = 0;
	float dd_dv12 = 0;
	float dd_dv13 = 0;
	float dd_dv14 = 0;
	// end
	
	// build __shared__ variables : __shared__ variables can be accessed and manipulated by all threads in the same block (ps: nedd to sync threads)
	// important attributes belonging to each related gaussian , to compute gradient w.r.t pose
	__shared__ int collected_id[BLOCK_SIZE];
	

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1]; //start in the BACK
			collected_id[block.thread_rank()] = coll_id;
		}
		block.sync();


		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			const int global_id = collected_id[j];
			
			if ((global_id == single_pix_gs_id[valid_seq])&(!shut))
			{	
				// Part-1 : color w.r.t Pose
				//Cam-x 
				glm::vec3 dp_dCx = single_dpixel_dgc[valid_seq] * dgc_dCampos[global_id * 3 + 0];
				//Cam-y
				glm::vec3 dp_dCy = single_dpixel_dgc[valid_seq] * dgc_dCampos[global_id * 3 + 1];
				//Cam-z
				glm::vec3 dp_dCz = single_dpixel_dgc[valid_seq] * dgc_dCampos[global_id * 3 + 2];

				// further : v0
				glm::vec3 dp_dv0_part1 = dp_dCx * (-viewmatrix[12]);
				// further : v1
				glm::vec3 dp_dv1_part1 = dp_dCx * (-viewmatrix[13]);
				// further : v2
				glm::vec3 dp_dv2_part1 = dp_dCx * (-viewmatrix[14]);
				// further : v4
				glm::vec3 dp_dv4_part1 = dp_dCy * (-viewmatrix[12]);
				// further : v5
				glm::vec3 dp_dv5_part1 = dp_dCy * (-viewmatrix[13]);
				// further : v6
				glm::vec3 dp_dv6_part1 = dp_dCy * (-viewmatrix[14]);
				// further : v8
				glm::vec3 dp_dv8_part1 = dp_dCz * (-viewmatrix[12]);
				// further : v9
				glm::vec3 dp_dv9_part1 = dp_dCz * (-viewmatrix[13]);
				// further : v10
				glm::vec3 dp_dv10_part1 = dp_dCz * (-viewmatrix[14]);

				// further : v12
				glm::vec3 dp_dv12_part1 = dp_dCx * (-viewmatrix[0]) + dp_dCy * (-viewmatrix[4]) + dp_dCz * (-viewmatrix[8]);
				// further : v13
				glm::vec3 dp_dv13_part1 = dp_dCx * (-viewmatrix[1]) + dp_dCy * (-viewmatrix[5]) + dp_dCz * (-viewmatrix[9]);
				// further : v14
				glm::vec3 dp_dv14_part1 = dp_dCx * (-viewmatrix[2]) + dp_dCy * (-viewmatrix[6]) + dp_dCz * (-viewmatrix[10]);



				// Part-2-1 : opacity w.r.t Pose
				// v0
				float3 dp_dv0_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 0].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 0].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv0_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 0].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 0].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// v1
				float3 dp_dv1_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 1].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 1].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv1_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 1].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 1].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv1_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 1].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 1].y;
				// v2
				float3 dp_dv2_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 2].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 2].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv2_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 2].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 2].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv2_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 2].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 2].y;
				// v4
				float3 dp_dv4_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 3].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 3].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv4_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 3].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 3].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv4_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 3].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 3].y;
				// v5
				float3 dp_dv5_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 4].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 4].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv5_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 4].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 4].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv5_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 4].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 4].y;
				// v6
				float3 dp_dv6_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 5].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 5].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv6_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 5].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 5].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv6_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 5].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 5].y;
				// v8
				float3 dp_dv8_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 6].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 6].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv8_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 6].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 6].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv8_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 6].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 6].y;
				// v9
				float3 dp_dv9_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 7].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 7].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv9_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 7].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 7].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv9_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 7].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 7].y;
				// v10
				float3 dp_dv10_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 8].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 8].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv10_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 8].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 8].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv10_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 8].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 8].y; 
				// v12
				float3 dp_dv12_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 9].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 9].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv12_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 9].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 9].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv12_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 9].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 9].y;
				// v13
				float3 dp_dv13_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 10].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 10].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv13_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 10].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 10].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv13_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 10].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 10].y; 
				// v14
				float3 dp_dv14_part2_1 = add_f3_f3(mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 11].x, single_dpixel_dndcs[valid_seq * 2 +0]), mul_f_f3(dgndcs_dviewmatrix[global_id * 12 + 11].y, single_dpixel_dndcs[valid_seq * 2 +1])); 
				float dd_dv14_part2_1 = dgndcs_dviewmatrix[global_id * 12 + 11].x * single_ddepth_dndcs[valid_seq * 2 +0] + dgndcs_dviewmatrix[global_id * 12 + 11].y * single_ddepth_dndcs[valid_seq * 2 +1];
				// float3 dp_dv14_part2_1 = single_dpixel_dndcs[valid_seq * 2 +0] * dgndcs_dviewmatrix[global_id * 12 + 11].x + single_dpixel_dndcs[valid_seq * 2 +1] * dgndcs_dviewmatrix[global_id * 12 + 11].y;

				// Part-2-2 : opacity w.r.t Pose
				float3 dp_dT00 = triple_add_f3(mul_f_f3(dginvcovs_dT[global_id * 6 + 0].x, single_dpixel_dinvcovs[valid_seq * 3 + 0]), mul_f_f3(dginvcovs_dT[global_id * 6 + 0].y, single_dpixel_dinvcovs[valid_seq * 3 + 1]), mul_f_f3(dginvcovs_dT[global_id * 6 + 0].z, single_dpixel_dinvcovs[valid_seq * 3 + 2]));
				float dd_dT00 = dginvcovs_dT[global_id * 6 + 0].x * single_ddepth_dinvcovs[valid_seq * 3 + 0] +  dginvcovs_dT[global_id * 6 + 0].y * single_ddepth_dinvcovs[valid_seq * 3 + 1] + dginvcovs_dT[global_id * 6 + 0].z * single_ddepth_dinvcovs[valid_seq * 3 + 2];
				// float3 dp_dT00 = single_dpixel_dinvcovs[valid_seq * 3 + 0] * dginvcovs_dT[global_id * 6 + 0].x + single_dpixel_dinvcovs[valid_seq * 3 + 1] * dginvcovs_dT[global_id * 6 + 0].y + single_dpixel_dinvcovs[valid_seq * 3 + 2] * dginvcovs_dT[global_id * 6 + 0].z;
				float3 dp_dT01 = triple_add_f3(mul_f_f3(dginvcovs_dT[global_id * 6 + 1].x, single_dpixel_dinvcovs[valid_seq * 3 + 0]), mul_f_f3(dginvcovs_dT[global_id * 6 + 1].y, single_dpixel_dinvcovs[valid_seq * 3 + 1]), mul_f_f3(dginvcovs_dT[global_id * 6 + 1].z, single_dpixel_dinvcovs[valid_seq * 3 + 2]));				
				float dd_dT01 = dginvcovs_dT[global_id * 6 + 1].x * single_ddepth_dinvcovs[valid_seq * 3 + 0] +  dginvcovs_dT[global_id * 6 + 1].y * single_ddepth_dinvcovs[valid_seq * 3 + 1] + dginvcovs_dT[global_id * 6 + 1].z * single_ddepth_dinvcovs[valid_seq * 3 + 2];
				// float3 dp_dT01 = single_dpixel_dinvcovs[valid_seq * 3 + 0] * dginvcovs_dT[global_id * 6 + 1].x + single_dpixel_dinvcovs[valid_seq * 3 + 1] * dginvcovs_dT[global_id * 6 + 1].y + single_dpixel_dinvcovs[valid_seq * 3 + 2] * dginvcovs_dT[global_id * 6 + 1].z;
				float3 dp_dT02 = triple_add_f3(mul_f_f3(dginvcovs_dT[global_id * 6 + 2].x, single_dpixel_dinvcovs[valid_seq * 3 + 0]), mul_f_f3(dginvcovs_dT[global_id * 6 + 2].y, single_dpixel_dinvcovs[valid_seq * 3 + 1]), mul_f_f3(dginvcovs_dT[global_id * 6 + 2].z, single_dpixel_dinvcovs[valid_seq * 3 + 2]));				
				float dd_dT02 = dginvcovs_dT[global_id * 6 + 2].x * single_ddepth_dinvcovs[valid_seq * 3 + 0] +  dginvcovs_dT[global_id * 6 + 2].y * single_ddepth_dinvcovs[valid_seq * 3 + 1] + dginvcovs_dT[global_id * 6 + 2].z * single_ddepth_dinvcovs[valid_seq * 3 + 2];
				// float3 dp_dT02 = single_dpixel_dinvcovs[valid_seq * 3 + 0] * dginvcovs_dT[global_id * 6 + 2].x + single_dpixel_dinvcovs[valid_seq * 3 + 1] * dginvcovs_dT[global_id * 6 + 2].y + single_dpixel_dinvcovs[valid_seq * 3 + 2] * dginvcovs_dT[global_id * 6 + 2].z;
				float3 dp_dT10 = triple_add_f3(mul_f_f3(dginvcovs_dT[global_id * 6 + 3].x, single_dpixel_dinvcovs[valid_seq * 3 + 0]), mul_f_f3(dginvcovs_dT[global_id * 6 + 3].y, single_dpixel_dinvcovs[valid_seq * 3 + 1]), mul_f_f3(dginvcovs_dT[global_id * 6 + 3].z, single_dpixel_dinvcovs[valid_seq * 3 + 2]));				
				float dd_dT10 = dginvcovs_dT[global_id * 6 + 3].x * single_ddepth_dinvcovs[valid_seq * 3 + 0] +  dginvcovs_dT[global_id * 6 + 3].y * single_ddepth_dinvcovs[valid_seq * 3 + 1] + dginvcovs_dT[global_id * 6 + 3].z * single_ddepth_dinvcovs[valid_seq * 3 + 2];
				// float3 dp_dT10 = single_dpixel_dinvcovs[valid_seq * 3 + 0] * dginvcovs_dT[global_id * 6 + 3].x + single_dpixel_dinvcovs[valid_seq * 3 + 1] * dginvcovs_dT[global_id * 6 + 3].y + single_dpixel_dinvcovs[valid_seq * 3 + 2] * dginvcovs_dT[global_id * 6 + 3].z;
				float3 dp_dT11 = triple_add_f3(mul_f_f3(dginvcovs_dT[global_id * 6 + 4].x, single_dpixel_dinvcovs[valid_seq * 3 + 0]), mul_f_f3(dginvcovs_dT[global_id * 6 + 4].y, single_dpixel_dinvcovs[valid_seq * 3 + 1]), mul_f_f3(dginvcovs_dT[global_id * 6 + 4].z, single_dpixel_dinvcovs[valid_seq * 3 + 2]));				
				float dd_dT11 = dginvcovs_dT[global_id * 6 + 4].x * single_ddepth_dinvcovs[valid_seq * 3 + 0] +  dginvcovs_dT[global_id * 6 + 4].y * single_ddepth_dinvcovs[valid_seq * 3 + 1] + dginvcovs_dT[global_id * 6 + 4].z * single_ddepth_dinvcovs[valid_seq * 3 + 2];
				// float3 dp_dT11 = single_dpixel_dinvcovs[valid_seq * 3 + 0] * dginvcovs_dT[global_id * 6 + 4].x + single_dpixel_dinvcovs[valid_seq * 3 + 1] * dginvcovs_dT[global_id * 6 + 4].y + single_dpixel_dinvcovs[valid_seq * 3 + 2] * dginvcovs_dT[global_id * 6 + 4].z;
				float3 dp_dT12 = triple_add_f3(mul_f_f3(dginvcovs_dT[global_id * 6 + 5].x, single_dpixel_dinvcovs[valid_seq * 3 + 0]), mul_f_f3(dginvcovs_dT[global_id * 6 + 5].y, single_dpixel_dinvcovs[valid_seq * 3 + 1]), mul_f_f3(dginvcovs_dT[global_id * 6 + 5].z, single_dpixel_dinvcovs[valid_seq * 3 + 2]));			
				float dd_dT12 = dginvcovs_dT[global_id * 6 + 5].x * single_ddepth_dinvcovs[valid_seq * 3 + 0] +  dginvcovs_dT[global_id * 6 + 5].y * single_ddepth_dinvcovs[valid_seq * 3 + 1] + dginvcovs_dT[global_id * 6 + 5].z * single_ddepth_dinvcovs[valid_seq * 3 + 2];
				// float3 dp_dT12 = single_dpixel_dinvcovs[valid_seq * 3 + 0] * dginvcovs_dT[global_id * 6 + 5].x + single_dpixel_dinvcovs[valid_seq * 3 + 1] * dginvcovs_dT[global_id * 6 + 5].y + single_dpixel_dinvcovs[valid_seq * 3 + 2] * dginvcovs_dT[global_id * 6 + 5].z;
				
				float3 G_world_center = means3D[global_id];
				float3 G_cam_center = transformPoint4x3(means3D[global_id], viewmatrix);
				float G_cam_center_z2 = G_cam_center.z * G_cam_center.z;

				float f_x_frac_z = f_x * (1.0f / G_cam_center.z);
				float f_y_frac_z = f_y * (1.0f / G_cam_center.z);
				float wx_camz = G_world_center.x / G_cam_center.z;
				float wy_camz = G_world_center.y / G_cam_center.z;
				float wz_camz = G_world_center.z / G_cam_center.z;
				float one_camz = 1.0f / G_cam_center.z;
				float camx_camz = G_cam_center.x / G_cam_center.z;
				float camy_camz = G_cam_center.y / G_cam_center.z;
				float camx_x_wx_camzz = G_cam_center.x * G_world_center.x / G_cam_center_z2;
				float camx_x_wy_camzz = G_cam_center.x * G_world_center.y / G_cam_center_z2;
				float camx_x_wz_camzz = G_cam_center.x * G_world_center.z / G_cam_center_z2;
				float camx_x_one_camzz = G_cam_center.x * 1.0f / G_cam_center_z2;
				float camy_x_wx_camzz = G_cam_center.y * G_world_center.x / G_cam_center_z2;
				float camy_x_wy_camzz = G_cam_center.y * G_world_center.y / G_cam_center_z2;
				float camy_x_wz_camzz = G_cam_center.y * G_world_center.z / G_cam_center_z2;
				float camy_x_one_camzz = G_cam_center.y * 1.0f / G_cam_center_z2;




				float dT00_dv0 = f_x_frac_z * (1.0f - viewmatrix[2] * wx_camz);
				float dT00_dv4 = f_x_frac_z * (-viewmatrix[2] * wy_camz);
				float dT00_dv8 = f_x_frac_z * (-viewmatrix[2] * wz_camz);
				float dT00_dv12 = f_x_frac_z * (-viewmatrix[2] * one_camz);
				float dT00_dv2 = f_x_frac_z * (2 * viewmatrix[2] * camx_x_wx_camzz - viewmatrix[0] * wx_camz - camx_camz);
				float dT00_dv6 = f_x_frac_z * (2 * viewmatrix[2] * camx_x_wy_camzz - viewmatrix[0] * wy_camz);
				float dT00_dv10 = f_x_frac_z * (2 * viewmatrix[2] * camx_x_wz_camzz - viewmatrix[0] * wz_camz);
				float dT00_dv14 = f_x_frac_z * (2 * viewmatrix[2] * camx_x_one_camzz - viewmatrix[0] * one_camz);
				float dT00_dv1 = 0;
				float dT00_dv5 = 0;
				float dT00_dv9 = 0;
				float dT00_dv13 = 0;

				float dT01_dv0 = f_x_frac_z * (-viewmatrix[6] * wx_camz);
				float dT01_dv4 = f_x_frac_z * (1.0f - viewmatrix[6] * wy_camz);
				float dT01_dv8 = f_x_frac_z * (-viewmatrix[6] * wz_camz);
				float dT01_dv12 = f_x_frac_z * (-viewmatrix[6] * one_camz);
				float dT01_dv2 = f_x_frac_z * (2 * viewmatrix[6] * camx_x_wx_camzz - viewmatrix[4] * wx_camz);
				float dT01_dv6 = f_x_frac_z * (2 * viewmatrix[6] * camx_x_wy_camzz - viewmatrix[4] * wy_camz - camx_camz);
				float dT01_dv10 = f_x_frac_z * (2 * viewmatrix[6] * camx_x_wz_camzz - viewmatrix[4] * wz_camz);
				float dT01_dv14 = f_x_frac_z * (2 * viewmatrix[6] * camx_x_one_camzz - viewmatrix[4] * one_camz);
				float dT01_dv1 = 0;
				float dT01_dv5 = 0;
				float dT01_dv9 = 0;
				float dT01_dv13 = 0;

				float dT02_dv0 = f_x_frac_z * (-viewmatrix[10] * wx_camz);
				float dT02_dv4 = f_x_frac_z * (-viewmatrix[10] * wy_camz);
				float dT02_dv8 = f_x_frac_z * (1.0f - viewmatrix[10] * wz_camz);
				float dT02_dv12 = f_x_frac_z * (-viewmatrix[10] * one_camz);
				float dT02_dv2 = f_x_frac_z * (2 * viewmatrix[10] * camx_x_wx_camzz - viewmatrix[8] * wx_camz);
				float dT02_dv6 = f_x_frac_z * (2 * viewmatrix[10] * camx_x_wy_camzz - viewmatrix[8] * wy_camz);
				float dT02_dv10 = f_x_frac_z * (2 * viewmatrix[10] * camx_x_wz_camzz - viewmatrix[8] * wz_camz - camx_camz);
				float dT02_dv14 = f_x_frac_z * (2 * viewmatrix[10] * camx_x_one_camzz - viewmatrix[8] * one_camz);
				float dT02_dv1 = 0;
				float dT02_dv5 = 0;
				float dT02_dv9 = 0;
				float dT02_dv13 = 0;

				float dT10_dv1 = f_y_frac_z * (1.0f - viewmatrix[2] * wx_camz);
				float dT10_dv5 = f_y_frac_z * (-viewmatrix[2] * wy_camz);
				float dT10_dv9 = f_y_frac_z * (-viewmatrix[2] * wz_camz);
				float dT10_dv13 = f_y_frac_z * (-viewmatrix[2] * one_camz);
				float dT10_dv2 = f_y_frac_z * (2 * viewmatrix[2] * camy_x_wx_camzz - viewmatrix[1] * wx_camz - camy_camz);
				float dT10_dv6 = f_y_frac_z * (2 * viewmatrix[2] * camy_x_wy_camzz - viewmatrix[1] * wy_camz);
				float dT10_dv10 = f_y_frac_z * (2 * viewmatrix[2] * camy_x_wz_camzz - viewmatrix[1] * wz_camz);
				float dT10_dv14 = f_y_frac_z * (2 * viewmatrix[2] * camy_x_one_camzz - viewmatrix[1] * one_camz);
				float dT10_dv0 = 0;
				float dT10_dv4 = 0;
				float dT10_dv8 = 0;
				float dT10_dv12 = 0;

				float dT11_dv1 = f_y_frac_z * (-viewmatrix[6] * wx_camz);
				float dT11_dv5 = f_y_frac_z * (1.0f - viewmatrix[6] * wy_camz);
				float dT11_dv9 = f_y_frac_z * (-viewmatrix[6] * wz_camz);
				float dT11_dv13 = f_y_frac_z * (-viewmatrix[6] * one_camz);
				float dT11_dv2 = f_y_frac_z * (2 * viewmatrix[6] * camy_x_wx_camzz - viewmatrix[5] * wx_camz);
				float dT11_dv6 = f_y_frac_z * (2 * viewmatrix[6] * camy_x_wy_camzz - viewmatrix[5] * wy_camz - camy_camz);
				float dT11_dv10 = f_y_frac_z * (2 * viewmatrix[6] * camy_x_wz_camzz - viewmatrix[5] * wz_camz);
				float dT11_dv14 = f_y_frac_z * (2 * viewmatrix[6] * camy_x_one_camzz - viewmatrix[5] * one_camz);
				float dT11_dv0 = 0;
				float dT11_dv4 = 0;
				float dT11_dv8 = 0;
				float dT11_dv12 = 0;

				float dT12_dv1 = f_y_frac_z * (-viewmatrix[10] * wx_camz);
				float dT12_dv5 = f_y_frac_z * (-viewmatrix[10] * wy_camz);
				float dT12_dv9 = f_y_frac_z * (1.0f - viewmatrix[10] * wz_camz);
				float dT12_dv13 = f_y_frac_z * (-viewmatrix[10] * one_camz);
				float dT12_dv2 = f_y_frac_z * (2 * viewmatrix[10] * camy_x_wx_camzz - viewmatrix[9] * wx_camz);
				float dT12_dv6 = f_y_frac_z * (2 * viewmatrix[10] * camy_x_wy_camzz - viewmatrix[9] * wy_camz);
				float dT12_dv10 = f_y_frac_z * (2 * viewmatrix[10] * camy_x_wz_camzz - viewmatrix[9] * wz_camz - camy_camz);
				float dT12_dv14 = f_y_frac_z * (2 * viewmatrix[10] * camy_x_one_camzz - viewmatrix[9] * one_camz);
				float dT12_dv0 = 0;
				float dT12_dv4 = 0;
				float dT12_dv8 = 0;
				float dT12_dv12 = 0;

				// v0 
				float3 dp_dv0_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv0, dp_dT00), mul_f_f3(dT01_dv0, dp_dT01), mul_f_f3(dT02_dv0, dp_dT02), mul_f_f3(dT10_dv0, dp_dT10), mul_f_f3(dT11_dv0, dp_dT11), mul_f_f3(dT12_dv0, dp_dT12)); 
				float dd_dv0_part2_2 = dT00_dv0 * dd_dT00 + dT01_dv0 * dd_dT01 + dT02_dv0 * dd_dT02 + dT10_dv0 * dd_dT10 + dT11_dv0 * dd_dT11 + dT12_dv0 * dd_dT12;
				// float3 dp_dv0_part2_2 = dp_dT00 * dT00_dv0 + dp_dT01 * dT01_dv0 + dp_dT02 * dT02_dv0 + dp_dT10 * dT10_dv0 + dp_dT11 * dT11_dv0 + dp_dT12 * dT12_dv0;
				// v1 
				float3 dp_dv1_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv1, dp_dT00), mul_f_f3(dT01_dv1, dp_dT01), mul_f_f3(dT02_dv1, dp_dT02), mul_f_f3(dT10_dv1, dp_dT10), mul_f_f3(dT11_dv1, dp_dT11), mul_f_f3(dT12_dv1, dp_dT12)); 
				float dd_dv1_part2_2 = dT00_dv1 * dd_dT00 + dT01_dv1 * dd_dT01 + dT02_dv1 * dd_dT02 + dT10_dv1 * dd_dT10 + dT11_dv1 * dd_dT11 + dT12_dv1 * dd_dT12;
				// float3 dp_dv1_part2_2 = dp_dT00 * dT00_dv1 + dp_dT01 * dT01_dv1 + dp_dT02 * dT02_dv1 + dp_dT10 * dT10_dv1 + dp_dT11 * dT11_dv1 + dp_dT12 * dT12_dv1;
				// v2 
				float3 dp_dv2_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv2, dp_dT00), mul_f_f3(dT01_dv2, dp_dT01), mul_f_f3(dT02_dv2, dp_dT02), mul_f_f3(dT10_dv2, dp_dT10), mul_f_f3(dT11_dv2, dp_dT11), mul_f_f3(dT12_dv2, dp_dT12)); 
				float dd_dv2_part2_2 = dT00_dv2 * dd_dT00 + dT01_dv2 * dd_dT01 + dT02_dv2 * dd_dT02 + dT10_dv2 * dd_dT10 + dT11_dv2 * dd_dT11 + dT12_dv2 * dd_dT12;
				// float3 dp_dv2_part2_2 = dp_dT00 * dT00_dv2 + dp_dT01 * dT01_dv2 + dp_dT02 * dT02_dv2 + dp_dT10 * dT10_dv2 + dp_dT11 * dT11_dv2 + dp_dT12 * dT12_dv2;
				// v4 
				float3 dp_dv4_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv4, dp_dT00), mul_f_f3(dT01_dv4, dp_dT01), mul_f_f3(dT02_dv4, dp_dT02), mul_f_f3(dT10_dv4, dp_dT10), mul_f_f3(dT11_dv4, dp_dT11), mul_f_f3(dT12_dv4, dp_dT12)); 
				float dd_dv4_part2_2 = dT00_dv4 * dd_dT00 + dT01_dv4 * dd_dT01 + dT02_dv4 * dd_dT02 + dT10_dv4 * dd_dT10 + dT11_dv4 * dd_dT11 + dT12_dv4 * dd_dT12;
				// float3 dp_dv4_part2_2 = dp_dT00 * dT00_dv4 + dp_dT01 * dT01_dv4 + dp_dT02 * dT02_dv4 + dp_dT10 * dT10_dv4 + dp_dT11 * dT11_dv4 + dp_dT12 * dT12_dv4;
				// v5 
				float3 dp_dv5_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv5, dp_dT00), mul_f_f3(dT01_dv5, dp_dT01), mul_f_f3(dT02_dv5, dp_dT02), mul_f_f3(dT10_dv5, dp_dT10), mul_f_f3(dT11_dv5, dp_dT11), mul_f_f3(dT12_dv5, dp_dT12)); 
				float dd_dv5_part2_2 = dT00_dv5 * dd_dT00 + dT01_dv5 * dd_dT01 + dT02_dv5 * dd_dT02 + dT10_dv5 * dd_dT10 + dT11_dv5 * dd_dT11 + dT12_dv5 * dd_dT12;
				// float3 dp_dv5_part2_2 = dp_dT00 * dT00_dv5 + dp_dT01 * dT01_dv5 + dp_dT02 * dT02_dv5 + dp_dT10 * dT10_dv5 + dp_dT11 * dT11_dv5 + dp_dT12 * dT12_dv5;
				// v6 
				float3 dp_dv6_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv6, dp_dT00), mul_f_f3(dT01_dv6, dp_dT01), mul_f_f3(dT02_dv6, dp_dT02), mul_f_f3(dT10_dv6, dp_dT10), mul_f_f3(dT11_dv6, dp_dT11), mul_f_f3(dT12_dv6, dp_dT12)); 
				float dd_dv6_part2_2 = dT00_dv6 * dd_dT00 + dT01_dv6 * dd_dT01 + dT02_dv6 * dd_dT02 + dT10_dv6 * dd_dT10 + dT11_dv6 * dd_dT11 + dT12_dv6 * dd_dT12;
				// float3 dp_dv6_part2_2 = dp_dT00 * dT00_dv6 + dp_dT01 * dT01_dv6 + dp_dT02 * dT02_dv6 + dp_dT10 * dT10_dv6 + dp_dT11 * dT11_dv6 + dp_dT12 * dT12_dv6;
				// v8 
				float3 dp_dv8_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv8, dp_dT00), mul_f_f3(dT01_dv8, dp_dT01), mul_f_f3(dT02_dv8, dp_dT02), mul_f_f3(dT10_dv8, dp_dT10), mul_f_f3(dT11_dv8, dp_dT11), mul_f_f3(dT12_dv8, dp_dT12)); 
				float dd_dv8_part2_2 = dT00_dv8 * dd_dT00 + dT01_dv8 * dd_dT01 + dT02_dv8 * dd_dT02 + dT10_dv8 * dd_dT10 + dT11_dv8 * dd_dT11 + dT12_dv8 * dd_dT12;
				// float3 dp_dv8_part2_2 = dp_dT00 * dT00_dv8 + dp_dT01 * dT01_dv8 + dp_dT02 * dT02_dv8 + dp_dT10 * dT10_dv8 + dp_dT11 * dT11_dv8 + dp_dT12 * dT12_dv8;
				// v9 
				float3 dp_dv9_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv9, dp_dT00), mul_f_f3(dT01_dv9, dp_dT01), mul_f_f3(dT02_dv9, dp_dT02), mul_f_f3(dT10_dv9, dp_dT10), mul_f_f3(dT11_dv9, dp_dT11), mul_f_f3(dT12_dv9, dp_dT12)); 
				float dd_dv9_part2_2 = dT00_dv9 * dd_dT00 + dT01_dv9 * dd_dT01 + dT02_dv9 * dd_dT02 + dT10_dv9 * dd_dT10 + dT11_dv9 * dd_dT11 + dT12_dv9 * dd_dT12;
				// float3 dp_dv9_part2_2 = dp_dT00 * dT00_dv9 + dp_dT01 * dT01_dv9 + dp_dT02 * dT02_dv9 + dp_dT10 * dT10_dv9 + dp_dT11 * dT11_dv9 + dp_dT12 * dT12_dv9;
				// v10 
				float3 dp_dv10_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv10, dp_dT00), mul_f_f3(dT01_dv10, dp_dT01), mul_f_f3(dT02_dv10, dp_dT02), mul_f_f3(dT10_dv10, dp_dT10), mul_f_f3(dT11_dv10, dp_dT11), mul_f_f3(dT12_dv10, dp_dT12)); 
				float dd_dv10_part2_2 = dT00_dv10 * dd_dT00 + dT01_dv10 * dd_dT01 + dT02_dv10 * dd_dT02 + dT10_dv10 * dd_dT10 + dT11_dv10 * dd_dT11 + dT12_dv10 * dd_dT12;
				// float3 dp_dv10_part2_2 = dp_dT00 * dT00_dv10 + dp_dT01 * dT01_dv10 + dp_dT02 * dT02_dv10 + dp_dT10 * dT10_dv10 + dp_dT11 * dT11_dv10 + dp_dT12 * dT12_dv10;
				// v12 
				float3 dp_dv12_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv12, dp_dT00), mul_f_f3(dT01_dv12, dp_dT01), mul_f_f3(dT02_dv12, dp_dT02), mul_f_f3(dT10_dv12, dp_dT10), mul_f_f3(dT11_dv12, dp_dT11), mul_f_f3(dT12_dv12, dp_dT12)); 
				float dd_dv12_part2_2 = dT00_dv12 * dd_dT00 + dT01_dv12 * dd_dT01 + dT02_dv12 * dd_dT02 + dT10_dv12 * dd_dT10 + dT11_dv12 * dd_dT11 + dT12_dv12 * dd_dT12;
				// float3 dp_dv12_part2_2 = dp_dT00 * dT00_dv12 + dp_dT01 * dT01_dv12 + dp_dT02 * dT02_dv12 + dp_dT10 * dT10_dv12 + dp_dT11 * dT11_dv12 + dp_dT12 * dT12_dv12;
				// v13 
				float3 dp_dv13_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv13, dp_dT00), mul_f_f3(dT01_dv13, dp_dT01), mul_f_f3(dT02_dv13, dp_dT02), mul_f_f3(dT10_dv13, dp_dT10), mul_f_f3(dT11_dv13, dp_dT11), mul_f_f3(dT12_dv13, dp_dT12)); 
				float dd_dv13_part2_2 = dT00_dv13 * dd_dT00 + dT01_dv13 * dd_dT01 + dT02_dv13 * dd_dT02 + dT10_dv13 * dd_dT10 + dT11_dv13 * dd_dT11 + dT12_dv13 * dd_dT12;
				// float3 dp_dv13_part2_2 = dp_dT00 * dT00_dv13 + dp_dT01 * dT01_dv13 + dp_dT02 * dT02_dv13 + dp_dT10 * dT10_dv13 + dp_dT11 * dT11_dv13 + dp_dT12 * dT12_dv13;
				// v14 
				float3 dp_dv14_part2_2 = sixtimes_add_f3(mul_f_f3(dT00_dv14, dp_dT00), mul_f_f3(dT01_dv14, dp_dT01), mul_f_f3(dT02_dv14, dp_dT02), mul_f_f3(dT10_dv14, dp_dT10), mul_f_f3(dT11_dv14, dp_dT11), mul_f_f3(dT12_dv14, dp_dT12)); 
				float dd_dv14_part2_2 = dT00_dv14 * dd_dT00 + dT01_dv14 * dd_dT01 + dT02_dv14 * dd_dT02 + dT10_dv14 * dd_dT10 + dT11_dv14 * dd_dT11 + dT12_dv14 * dd_dT12;
				// float3 dp_dv14_part2_2 = dp_dT00 * dT00_dv14 + dp_dT01 * dT01_dv14 + dp_dT02 * dT02_dv14 + dp_dT10 * dT10_dv14 + dp_dT11 * dT11_dv14 + dp_dT12 * dT12_dv14;


				float dd_dv0_part1 = 0;
				float dd_dv1_part1 = 0;
				float dd_dv2_part1 = single_dpixel_dgc[valid_seq] * G_world_center.x;
				float dd_dv4_part1 = 0;
				float dd_dv5_part1 = 0;
				float dd_dv6_part1 = single_dpixel_dgc[valid_seq] * G_world_center.y;
				float dd_dv8_part1 = 0;
				float dd_dv9_part1 = 0;
				float dd_dv10_part1 = single_dpixel_dgc[valid_seq] * G_world_center.z;
				float dd_dv12_part1 = 0;
				float dd_dv13_part1 = 0;
				float dd_dv14_part1 = single_dpixel_dgc[valid_seq] * 1.0f;



				// Summary

				dp_dv0 = add_f3_f3(dp_dv0, add_f3_f3(float3{dp_dv0_part1.x, dp_dv0_part1.y, dp_dv0_part1.z}, dp_dv0_part2_1));
				dp_dv1 = add_f3_f3(dp_dv1, add_f3_f3(float3{dp_dv1_part1.x, dp_dv1_part1.y, dp_dv1_part1.z}, dp_dv1_part2_1));
				dp_dv2 = add_f3_f3(dp_dv2, add_f3_f3(float3{dp_dv2_part1.x, dp_dv2_part1.y, dp_dv2_part1.z}, dp_dv2_part2_1));
				dp_dv4 = add_f3_f3(dp_dv4, add_f3_f3(float3{dp_dv4_part1.x, dp_dv4_part1.y, dp_dv4_part1.z}, dp_dv4_part2_1));
				dp_dv5 = add_f3_f3(dp_dv5, add_f3_f3(float3{dp_dv5_part1.x, dp_dv5_part1.y, dp_dv5_part1.z}, dp_dv5_part2_1));
				dp_dv6 = add_f3_f3(dp_dv6, add_f3_f3(float3{dp_dv6_part1.x, dp_dv6_part1.y, dp_dv6_part1.z}, dp_dv6_part2_1));
				dp_dv8 = add_f3_f3(dp_dv8, add_f3_f3(float3{dp_dv8_part1.x, dp_dv8_part1.y, dp_dv8_part1.z}, dp_dv8_part2_1));
				dp_dv9 = add_f3_f3(dp_dv9, add_f3_f3(float3{dp_dv9_part1.x, dp_dv9_part1.y, dp_dv9_part1.z}, dp_dv9_part2_1));
				dp_dv10 = add_f3_f3(dp_dv10, add_f3_f3(float3{dp_dv10_part1.x, dp_dv10_part1.y, dp_dv10_part1.z}, dp_dv10_part2_1));
				dp_dv12 = add_f3_f3(dp_dv12, add_f3_f3(float3{dp_dv12_part1.x, dp_dv12_part1.y, dp_dv12_part1.z}, dp_dv12_part2_1));
				dp_dv13 = add_f3_f3(dp_dv13, add_f3_f3(float3{dp_dv13_part1.x, dp_dv13_part1.y, dp_dv13_part1.z}, dp_dv13_part2_1));
				dp_dv14 = add_f3_f3(dp_dv14, add_f3_f3(float3{dp_dv14_part1.x, dp_dv14_part1.y, dp_dv14_part1.z}, dp_dv14_part2_1));


				dd_dv0 = dd_dv0_part1 + dd_dv0_part2_1;
				dd_dv1 = dd_dv1_part1 + dd_dv1_part2_1;
				dd_dv2 = dd_dv2_part1 + dd_dv2_part2_1;
				dd_dv4 = dd_dv4_part1 + dd_dv4_part2_1;
				dd_dv5 = dd_dv5_part1 + dd_dv5_part2_1;
				dd_dv6 = dd_dv6_part1 + dd_dv6_part2_1;
				dd_dv8 = dd_dv8_part1 + dd_dv8_part2_1;
				dd_dv9 = dd_dv9_part1 + dd_dv9_part2_1;
				dd_dv10 = dd_dv10_part1 + dd_dv10_part2_1;
				dd_dv12 = dd_dv12_part1 + dd_dv12_part2_1;
				dd_dv13 = dd_dv13_part1 + dd_dv13_part2_1;
				dd_dv14 = dd_dv14_part1 + dd_dv14_part2_1;



				if (valid_seq == length-1)
				{
					shut = true;
					continue; 
				}
				valid_seq++;
			}
			else
			{
				continue;
			}
		}
	}

	float dL_dpixel[C];
	float dL_depth;
	for (int i = 0; i < C; i++)
		dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	dL_depth = dL_depths[pix_id];

	float dL_dv0 = dL_dpixel[0] * dp_dv0.x + dL_dpixel[1] * dp_dv0.y + dL_dpixel[2] * dp_dv0.z + dL_depth * dd_dv0; 
	float dL_dv1 = dL_dpixel[0] * dp_dv1.x + dL_dpixel[1] * dp_dv1.y + dL_dpixel[2] * dp_dv1.z + dL_depth * dd_dv1; 
	float dL_dv2 = dL_dpixel[0] * dp_dv2.x + dL_dpixel[1] * dp_dv2.y + dL_dpixel[2] * dp_dv2.z + dL_depth * dd_dv2; 
	float dL_dv4 = dL_dpixel[0] * dp_dv4.x + dL_dpixel[1] * dp_dv4.y + dL_dpixel[2] * dp_dv4.z + dL_depth * dd_dv4; 
	float dL_dv5 = dL_dpixel[0] * dp_dv5.x + dL_dpixel[1] * dp_dv5.y + dL_dpixel[2] * dp_dv5.z + dL_depth * dd_dv5; 
	float dL_dv6 = dL_dpixel[0] * dp_dv6.x + dL_dpixel[1] * dp_dv6.y + dL_dpixel[2] * dp_dv6.z + dL_depth * dd_dv6; 
	float dL_dv8 = dL_dpixel[0] * dp_dv8.x + dL_dpixel[1] * dp_dv8.y + dL_dpixel[2] * dp_dv8.z + dL_depth * dd_dv8; 
	float dL_dv9 = dL_dpixel[0] * dp_dv9.x + dL_dpixel[1] * dp_dv9.y + dL_dpixel[2] * dp_dv9.z + dL_depth * dd_dv9; 
	float dL_dv10 = dL_dpixel[0] * dp_dv10.x + dL_dpixel[1] * dp_dv10.y + dL_dpixel[2] * dp_dv10.z + dL_depth * dd_dv10; 
	float dL_dv12 = dL_dpixel[0] * dp_dv12.x + dL_dpixel[1] * dp_dv12.y + dL_dpixel[2] * dp_dv12.z + dL_depth * dd_dv12; 
	float dL_dv13 = dL_dpixel[0] * dp_dv13.x + dL_dpixel[1] * dp_dv13.y + dL_dpixel[2] * dp_dv13.z + dL_depth * dd_dv13; 
	float dL_dv14 = dL_dpixel[0] * dp_dv14.x + dL_dpixel[1] * dp_dv14.y + dL_dpixel[2] * dp_dv14.z + dL_depth * dd_dv14; 

	atomicAdd(&(dL_dview[0]), dL_dv0);
	atomicAdd(&(dL_dview[1]), dL_dv1);
	atomicAdd(&(dL_dview[2]), dL_dv2);
	atomicAdd(&(dL_dview[4]), dL_dv4);
	atomicAdd(&(dL_dview[5]), dL_dv5);
	atomicAdd(&(dL_dview[6]), dL_dv6);
	atomicAdd(&(dL_dview[8]), dL_dv8);
	atomicAdd(&(dL_dview[9]), dL_dv9);
	atomicAdd(&(dL_dview[10]), dL_dv10);
	atomicAdd(&(dL_dview[12]), dL_dv12);
	atomicAdd(&(dL_dview[13]), dL_dv13);
	atomicAdd(&(dL_dview[14]), dL_dv14);
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
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot, 
	float* dgc_dCampos, 
	const float* perspec_matrix,
	float* dgc_ndcs_dviewmatrix,
	float* dgc_invcovs_dT,
	float* dL_dgau_depth
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
		dL_dcov3D, 
		(float3*)dgc_invcovs_dT,
		dL_dgau_depth
		);

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
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot, 
		dgc_dCampos,
		perspec_matrix, 
		(float2*)dgc_ndcs_dviewmatrix
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
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors, 
	float* dpixel_dgc, 
	int* gau_id_list, 
	int* pix_id_list, 
	const uint32_t* n_valid_contrib_cumsum, 
    float* dpixel_dndcs, 
	float* dpixel_dinvcovs,
	float* dL_dgau_depths,
	float* ddepth_dndcs, 
	float* ddepth_dinvcovs,
	const float* gt_depth,
	const float* dL_duncertainties
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
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dpixel_dgc, 
		gau_id_list, 
		pix_id_list, 
		n_valid_contrib_cumsum,
		(float3*)dpixel_dndcs,
		(float3*)dpixel_dinvcovs,
		dL_dgau_depths,
		ddepth_dndcs,
		ddepth_dinvcovs,
		gt_depth,
		dL_duncertainties
		);
}

void BACKWARD::ComputePoseGrad(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float3* means,
	const float* view,
	const float focal_x, float focal_y,
	float* dpixel_dgc, 
	float* dgc_dCampos,
	int* gau_id_list, 
	int* pix_id_list, 
	const uint32_t* n_valid_contrib_cumsum, 
	const uint32_t* n_valid_contrib,
	float* dpixel_dndcs, 
	float* dgc_ndcs_dviewmatrix, 
	float* dpixel_dinvcovs, 
	float* dgc_invcovs_dT, 
	float* dL_dview,
	const float* dL_dpixels,
	float* ddepth_dndcs, 
	float* ddepth_dinvcovs,
	const float* dL_depths
)
{
	ComputePG<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		means, 
		view, 
		focal_x, focal_y, 
		dpixel_dgc, 
		(glm::vec3*)dgc_dCampos, 
		gau_id_list, 
		pix_id_list, 
		n_valid_contrib_cumsum, 
		n_valid_contrib, 
		(float3*)dpixel_dndcs, 
		(float2*)dgc_ndcs_dviewmatrix, 
		(float3*)dpixel_dinvcovs, 
		(float3*)dgc_invcovs_dT, 
		dL_dview,
		dL_dpixels,
		ddepth_dndcs,
		ddepth_dinvcovs,
		dL_depths
		);
}
