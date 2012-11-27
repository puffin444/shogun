/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * This code specifically adapted from infLaplace.m
 *
 */
#include <shogun/lib/config.h>
 
#ifdef HAVE_LAPACK
#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/VBInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/regression/gp/StudentsTLikelihood.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/external/brent.h>
#include <iostream>

using namespace shogun;
using namespace Eigen;
using namespace std;

namespace shogun
{
	/*Wrapper class used for the Brent minimizer
	 *
	 */
	class Psi_line2 : public func_base
	{
	public:

		Eigen::Map<Eigen::MatrixXd>* K;
		SGVector<float64_t>* mW;
		float64_t scale;
		CLikelihoodModel* lik;
		CRegressionLabels *sg_labels;
		Eigen::VectorXd* variance;
		Eigen::MatrixXd* ddga;
		SGVector<float64_t>* true_b;
		SGMatrix<float64_t>* chol;
		
//ga+s*ddga

		virtual double operator() (double x)
		{



			VectorXd temp_variance = *variance + x*(*ddga);

			SGVector<float64_t> sg_grad(temp_variance.rows());

			for (index_t i = 0; i < temp_variance.rows(); i++)
				sg_grad[i] = temp_variance(i);

			SGVector<float64_t> h = lik->get_h(sg_labels, sg_grad);
			SGVector<float64_t> b = lik->get_b(sg_labels, sg_grad);
			Map<VectorXd> eigen_b(b.vector, b.vlen);
			SGVector<float64_t> dh = lik->get_first_derivative_h(sg_labels, sg_grad);		Map<VectorXd> eigen_dh(dh.vector, dh.vlen);
			SGVector<float64_t> db = lik->get_first_derivative_b(sg_labels, sg_grad);		Map<VectorXd> eigen_db(db.vector, db.vlen);
			SGVector<float64_t> d2h = lik->get_second_derivative_h(sg_labels, sg_grad);		Map<VectorXd> eigen_d2h(d2h.vector, d2h.vlen);
			SGVector<float64_t> d2b = lik->get_second_derivative_b(sg_labels, sg_grad);		Map<VectorXd> eigen_d2b(d2b.vector, d2b.vlen);


			VectorXd eigen_W = VectorXd::Constant(variance->rows(), 1).cwiseQuotient(*variance);
			VectorXd eigen_sW(eigen_W.rows());

			for (index_t i = 0; i < eigen_W.rows(); i++)
				eigen_sW[i] = CMath::sqrt(eigen_W[i]);


			LLT<MatrixXd> L((eigen_sW*eigen_sW.transpose()).cwiseProduct(
				(*K)*scale*scale) +
				MatrixXd::Identity(K->rows(), K->cols()));

			MatrixXd temp2 = L.matrixL();
			MatrixXd temp3 = L.matrixL();

			MatrixXd temp1 = eigen_sW.rowwise().replicate(K->rows());
		
			MatrixXd C = temp3.colPivHouseholderQr().solve(temp1.cwiseProduct(
				((*K)*scale*scale)));

			VectorXd t = C*eigen_b;



			VectorXd temp = t.transpose()*t-eigen_b.transpose()*(*K)*(eigen_b); 

			VectorXd diag = temp3.diagonal();			

			double sum = 0;

			
			for (index_t i = 0; i < diag.rows(); i++)
			{
				sum += CMath::log(diag[i]);
				sum += h[i]/2.0;
			}
			
			sum += temp[0]/2.0;

			for (index_t i = 0; i < mW->vlen; i++)
			{
				(*mW)[i] = eigen_W[i];
				(*true_b)[i] = b[i];
			}


			for (index_t i = 0; i < chol->num_rows; i++)
			{
				for (index_t j = 0; j < chol->num_cols; j++)
					(*chol)(i,j) = temp3(i,j);
			}



			return sum;

		}


	};
}

CVBInferenceMethod::CVBInferenceMethod() : CInferenceMethod()
{
	init();
	update_all();
	update_parameter_hash();
}

CVBInferenceMethod::CVBInferenceMethod(CKernel* kern,
		CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod) :
		CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	update_all();
}

void CVBInferenceMethod::init()
{
	m_latent_features = NULL;
	m_max_itr = 30;
	m_opt_tolerance = 1e-4;
	m_tolerance = 1e-7;
	m_max = 5;
	smax = 5; 
	ep = 1e-8;
	m_lz = 0;
}

CVBInferenceMethod::~CVBInferenceMethod()
{
}

void CVBInferenceMethod::update_all()
{
	if (m_labels)
		m_label_vector =
				((CRegressionLabels*) m_labels)->get_labels().clone();

	if (m_features && m_features->has_property(FP_DOT)
			&& m_features->get_num_vectors())
	{
		m_feature_matrix =
				((CDotFeatures*)m_features)->get_computed_dot_feature_matrix();

	}

	else if (m_features && m_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* feat =
				(CDotFeatures*)((CCombinedFeatures*)m_features)->
				get_first_feature_obj();

		if (feat->get_num_vectors())
			m_feature_matrix = feat->get_computed_dot_feature_matrix();

		SG_UNREF(feat);
	}

	update_data_means();

	if (m_kernel)
		update_train_kernel();

	if (m_ktrtr.num_cols*m_ktrtr.num_rows)
	{
		update_alpha();
		update_chol();
	}
}

void CVBInferenceMethod::check_members()
{
	if (!m_labels)
		SG_ERROR("No labels set\n");

	if (m_labels->get_label_type() != LT_REGRESSION)
		SG_ERROR("Expected RegressionLabels\n");

	if (!m_features)
		SG_ERROR("No features set!\n");

	if (m_labels->get_num_labels() != m_features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n");

	if(m_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* feat =
				(CDotFeatures*)((CCombinedFeatures*)m_features)->
				get_first_feature_obj();

		if (!feat->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n");

		if (feat->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");

		if (feat->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");

		SG_UNREF(feat);
	}

	else
	{
		if (!m_features->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n");

		if (m_features->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");

		if (m_features->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");
	}

	if (!m_kernel)
		SG_ERROR( "No kernel assigned!\n");

	if (!m_mean)
		SG_ERROR( "No mean function assigned!\n");

}

CMap<TParameter*, SGVector<float64_t> > CVBInferenceMethod::
get_marginal_likelihood_derivatives(CMap<TParameter*,
		CSGObject*>& para_dict)
{
	check_members();



	if(update_parameter_hash())
		update_all();

	m_kernel->build_parameter_dictionary(para_dict);
	m_mean->build_parameter_dictionary(para_dict);
	m_model->build_parameter_dictionary(para_dict);

	m_b = m_model->get_b((CRegressionLabels*)m_labels, m_variances);
	MatrixXd iKtil(m_Ktil.num_rows, m_Ktil.num_cols);

	MatrixXd eigen_chol(m_L.num_rows, m_L.num_cols);

	Map<MatrixXd> eigen_temp_kernel(temp_kernel.matrix, 
        	temp_kernel.num_rows, temp_kernel.num_cols);

	for (index_t i = 0; i < m_L.num_rows; i++)
	{
		for (index_t j = 0; j < m_L.num_cols; j++)
			eigen_chol(i,j) = m_L(i,j);
	}

	VectorXd eigen_b(m_b.vlen);
	
	for (index_t i = 0; i < m_b.vlen; i++)
		eigen_b[i] = m_b[i];

	VectorXd eigen_W(W.vlen);
	
	for (index_t i = 0; i < W.vlen; i++)
		eigen_W[i] = W[i];

	VectorXd eigen_alpha(m_alpha.vlen);
	
	for (index_t i = 0; i < m_alpha.vlen; i++)
		eigen_alpha[i] = m_alpha[i];

	for (index_t i = 0; i < m_Ktil.num_rows; i++)
	{
		for (index_t j = 0; j < m_Ktil.num_cols; j++)
			iKtil(i,j) = m_Ktil(i,j);
	}

	CMap<TParameter*, SGVector<float64_t> > gradient(
			3+para_dict.get_num_elements(),
			3+para_dict.get_num_elements());

	VectorXd sum(1);

	for (index_t i = 0; i < para_dict.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, CSGObject*>* node =
				para_dict.get_node_ptr(i);

		TParameter* param = node->key;
		CSGObject* obj = node->data;

		index_t length = 1;

		if ((param->m_datatype.m_ctype== CT_VECTOR ||
				param->m_datatype.m_ctype == CT_SGVECTOR) &&
				param->m_datatype.m_length_y != NULL)
			length = *(param->m_datatype.m_length_y);

		SGVector<float64_t> variables(length);

		bool deriv_found = false;

		Map<VectorXd> eigen_temp_alpha(temp_alpha.vector,
			temp_alpha.vlen);

		for (index_t h = 0; h < length; h++)
		{
			sum[0] = 0;
			SGMatrix<float64_t> deriv;
			SGVector<float64_t> mean_derivatives;
			VectorXd mean_dev_temp;
			SGVector<float64_t> lik_first_deriv;

			if (param->m_datatype.m_ctype == CT_VECTOR ||
					param->m_datatype.m_ctype == CT_SGVECTOR)
			{
				deriv = m_kernel->get_parameter_gradient(param, obj);

				mean_derivatives = m_mean->get_parameter_derivative(
						param, obj, m_feature_matrix, h);

				lik_first_deriv = m_model->get_first_derivative_h_param((CRegressionLabels*)m_labels,param, obj, m_variances);

				for (index_t d = 0; d < mean_derivatives.vlen; d++)
					mean_dev_temp[d] = mean_derivatives[d];
			}

			else
			{
				mean_derivatives = m_mean->get_parameter_derivative(
						param, obj, m_feature_matrix);
				for (index_t d = 0; d < mean_derivatives.vlen; d++)
					mean_dev_temp[d] = mean_derivatives[d];

				deriv = m_kernel->get_parameter_gradient(param, obj);

				lik_first_deriv = m_model->get_first_derivative_h_param((CRegressionLabels*)m_labels,param, obj, m_variances);

			}

			if (deriv.num_cols*deriv.num_rows > 0)
			{
				MatrixXd dK(deriv.num_cols, deriv.num_rows);

				for (index_t d = 0; d < deriv.num_rows; d++)
				{
					for (index_t s = 0; s < deriv.num_cols; s++)
						dK(d,s) = deriv(d,s);
				}

				sum[0] = (iKtil.cwiseProduct(dK)).sum()/2.0;
	
				VectorXd v = iKtil*(eigen_b.cwiseProduct(eigen_W));

				sum[0] -= (v.transpose()*dK*v).sum()/2.0;

				variables[h] = sum[0];

				deriv_found = true;
			}

			else if (mean_derivatives.vlen > 0)
			{
				sum = -eigen_alpha.transpose()*mean_dev_temp;
				variables[h] = sum[0];
				deriv_found = true;
			}

			else if (lik_first_deriv[0] != CMath::INFTY)
			{

				if (m_model->get_model_type() == LT_GAUSSIAN)
				{
					MatrixXd temp1 = eigen_chol.colPivHouseholderQr().solve(MatrixXd::Identity(eigen_temp_kernel.rows(), eigen_temp_kernel.cols()));

					temp1 = temp1.cwiseProduct(temp1);

					sum[0] = temp1.sum();

		float64_t m_sigma = ((CGaussianLikelihood*)m_model)->get_sigma();

					sum[0] -= m_sigma*m_sigma*(eigen_alpha*eigen_alpha.transpose()).sum();

					variables[h] = sum[0];
					deriv_found = true;

				}

				else
				{
					sum[0] = 0;
 					for (index_t dd = 0; dd < lik_first_deriv.vlen; dd++)
						sum[0] += lik_first_deriv[dd]/2.0;
					variables[h] = sum[0];
					deriv_found = true;
				}			
			}

		}

		if (deriv_found)
			gradient.add(param, variables);

	}

	TParameter* param;
	index_t index = get_modsel_param_index("scale");
	param = m_model_selection_parameters->get_parameter(index);

	MatrixXd dK(m_ktrtr.num_cols, m_ktrtr.num_rows);

	for (index_t d = 0; d < m_ktrtr.num_rows; d++)
	{
		for (index_t s = 0; s < m_ktrtr.num_cols; s++)
			dK(d,s) = m_ktrtr(d,s)*m_scale*2.0;;
	}



	sum[0] = (iKtil.cwiseProduct(dK)).sum()/2.0;

	VectorXd v = iKtil*(eigen_b.cwiseProduct(eigen_W));

	sum[0] -= (v.transpose()*dK*v).sum()/2.0;

	SGVector<float64_t> scale(1);

	scale[0] = sum[0];

	gradient.add(param, scale);
	para_dict.add(param, this);

	return gradient;
}


SGVector<float64_t> CVBInferenceMethod::get_diagonal_vector()
{
	SGVector<float64_t> result(sW.vlen);

	for (index_t i = 0; i < sW.vlen; i++)
		result[i] = sW[i];

	return result;
}

float64_t CVBInferenceMethod::get_negative_marginal_likelihood()
{
 return m_lz;
}

SGVector<float64_t> CVBInferenceMethod::get_alpha()
{
	if(update_parameter_hash())
		update_all();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CVBInferenceMethod::get_cholesky()
{
	if(update_parameter_hash())
		update_all();

	SGMatrix<float64_t> result(m_L);
	return result;
}

void CVBInferenceMethod::update_train_kernel()
{
	m_kernel->cleanup();

	m_kernel->init(m_features, m_features);

	//K(X, X)
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();

	m_ktrtr=kernel_matrix.clone();

	temp_kernel =SGMatrix<float64_t>(kernel_matrix.num_rows, kernel_matrix.num_cols);

	for (index_t i = 0; i < kernel_matrix.num_rows; i++)
	{
		for (index_t j = 0; j < kernel_matrix.num_cols; j++)
			temp_kernel(i,j) = kernel_matrix(i,j);
	}
}


void CVBInferenceMethod::update_chol()
{
	check_members();
 
}

void CVBInferenceMethod::update_alpha()
{
	float64_t Psi_Old = CMath::INFTY;
	float64_t Psi_New = 1e100;
	float64_t Psi_Def;

	VectorXd variance(temp_kernel.num_cols);

	m_L = SGMatrix<float64_t>(temp_kernel.num_rows, temp_kernel.num_cols);

	Map<MatrixXd> eigen_temp_kernel(temp_kernel.matrix, temp_kernel.num_rows, temp_kernel.num_cols);

	m_Ktil = SGMatrix<float64_t>(temp_kernel.num_rows, temp_kernel.num_cols);

	if (m_model->get_model_type() == LT_GAUSSIAN)
	{
		variance = VectorXd::Constant(temp_kernel.num_cols, 1);
		float64_t m_sigma = ((CGaussianLikelihood*)m_model)->get_sigma();
		variance = variance*m_sigma*m_sigma;

	}
/*
  % INNER compute the Newton direction of ga
  ga = ones(n,1);                                               % initial values
  itinner = 0;
  nlZ_new = 1e100; nlZ_old = Inf;                  % make sure while loop starts
  while nlZ_old-nlZ_new>tol && itinner<maxitinner                 % begin Newton
    itinner = itinner+1;

  end*/

	else
	{
		variance = VectorXd::Constant(temp_kernel.num_cols, 1);

		SGVector<float64_t> sg_grad(variance.rows());

		for (index_t i = 0; i < variance.rows(); i++)
			sg_grad[i] = variance(i);

		int itr = 0;

		while (Psi_Old - Psi_New > m_tolerance && itr < m_max_itr)
		{
			itr++;
//   			 [nlZ_old,dga,d2ga] = Psi(ga,K,inf,hyp,lik,y);

			Psi_Old = Psi_New;

			CRegressionLabels* sg_labels = ((CRegressionLabels*)m_labels);
			SGVector<float64_t> h = m_model->get_h(sg_labels, sg_grad);
			SGVector<float64_t> b = m_model->get_b(sg_labels, sg_grad);
			Map<VectorXd> eigen_b(b.vector, b.vlen);
			SGVector<float64_t> dh = m_model->get_first_derivative_h(sg_labels, sg_grad);		Map<VectorXd> eigen_dh(dh.vector, dh.vlen);
			SGVector<float64_t> db = m_model->get_first_derivative_b(sg_labels, sg_grad);		Map<VectorXd> eigen_db(db.vector, db.vlen);
			SGVector<float64_t> d2h = m_model->get_second_derivative_h(sg_labels, sg_grad);		Map<VectorXd> eigen_d2h(d2h.vector, d2h.vlen);
			SGVector<float64_t> d2b = m_model->get_second_derivative_b(sg_labels, sg_grad);		Map<VectorXd> eigen_d2b(d2b.vector, d2b.vlen);
			
			VectorXd eigen_W = VectorXd::Constant(variance.rows(), 1).cwiseQuotient(variance);
			VectorXd eigen_sW(eigen_W.rows());

			for (index_t i = 0; i < eigen_W.rows(); i++)
				eigen_sW[i] = CMath::sqrt(eigen_W[i]);

			LLT<MatrixXd> L((eigen_sW*eigen_sW.transpose()).cwiseProduct(
				eigen_temp_kernel*m_scale*m_scale) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

			MatrixXd chol = L.matrixL();
			MatrixXd temp2 = L.matrixL();

			MatrixXd temp1 = eigen_sW.rowwise().replicate(eigen_temp_kernel.rows());
		
			MatrixXd C = chol.colPivHouseholderQr().solve(temp1.cwiseProduct(
				(eigen_temp_kernel*m_scale*m_scale)));

			VectorXd t = C*eigen_b;

			MatrixXd arg = eigen_sW.asDiagonal();
			chol = chol.colPivHouseholderQr().solve(arg);

			chol = temp2.transpose().colPivHouseholderQr().solve(chol);

			MatrixXd iKtil = temp1.cwiseProduct(chol);


			for(index_t i = 0; i < iKtil.rows(); i++)
			{
				for(index_t j = 0; j < iKtil.cols(); j++)
					m_Ktil(i,j) = iKtil(i,j);
			}

			MatrixXd Khat = eigen_temp_kernel - C.transpose()*C;
			
			MatrixXd v = Khat*eigen_b;

			VectorXd temp3 = v.cwiseProduct(variance);

			temp3 = temp3.cwiseProduct(temp3);

			MatrixXd dga = iKtil.diagonal() - eigen_W + eigen_dh - temp3 - 2*v.cwiseProduct(eigen_db);
			dga = dga/2.0;

			VectorXd little_w = v.cwiseQuotient(variance.cwiseProduct(variance));
		
			MatrixXd d2ga = -1*(iKtil.cwiseProduct(iKtil));

			MatrixXd temp5 = ((eigen_W.cwiseProduct(eigen_W)).asDiagonal());

		//	std::cout << "temp5" << temp5 << std::endl;
		//	std::cout << "m" << MatrixXd(eigen_d2h.asDiagonal()) << std::endl;
		//	std::cout <<  "f" << Khat.cwiseProduct(little_w*(little_w+2*eigen_db).transpose()) << std::endl;
		//	std::cout << "n5" << Khat.cwiseProduct(eigen_db*eigen_db.transpose())
		//	 << std::endl;

			d2ga = d2ga + temp5;
			d2ga = d2ga + MatrixXd(eigen_d2h.asDiagonal());
			d2ga = d2ga/2.0;
			d2ga = d2ga - Khat.cwiseProduct(little_w*(little_w+2*eigen_db).transpose());
			d2ga = d2ga - Khat.cwiseProduct(eigen_db*eigen_db.transpose());

			VectorXd temp6 = (little_w.cwiseProduct(little_w));

			temp6 = temp6.cwiseProduct(variance);

			d2ga = d2ga + MatrixXd(temp6.asDiagonal());
			d2ga = d2ga - MatrixXd((v.cwiseProduct(eigen_d2b)).asDiagonal());
			d2ga = (d2ga + d2ga.transpose())/2.0;


			temp1 = d2ga+ep*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);


			MatrixXd ddga = -1.0*(temp1.colPivHouseholderQr().solve(dga));

			
			float64_t g = -ddga.maxCoeff();

 			float64_t s = 0.99*(variance/CMath::max(g,0.0)).minCoeff(); 
			s = CMath::min(s,smax);
				
			Psi_line2 func;
			func.sg_labels = (CRegressionLabels*)m_labels;
			func.K = &eigen_temp_kernel;
			func.scale = m_scale;
		
			func.lik = m_model;
	
			func.mW = &W;

			func.variance = &variance;
			func.ddga = &ddga;

			func.true_b = &b;
			func.chol = &m_L;

			double x = 0;
			Psi_New = local_min(0, s, m_opt_tolerance, func, x);
  
    			variance = variance + x*ddga;  
			
			for (index_t i = 0; i < variance.rows(); i++)
				variance[i] = CMath::abs(variance[i]); 
		}
		
	}

		SGVector<float64_t> sg_grad(variance.rows());

		for (index_t i = 0; i < variance.rows(); i++)
			sg_grad[i] = variance(i);

			CRegressionLabels* sg_labels = ((CRegressionLabels*)m_labels);
			SGVector<float64_t> b = m_model->get_b(sg_labels, sg_grad);
			Map<VectorXd> eigen_b(b.vector, b.vlen);
			
			VectorXd eigen_W = VectorXd::Constant(variance.rows(), 1).cwiseQuotient(variance);
			VectorXd eigen_sW(eigen_W.rows());

			for (index_t i = 0; i < eigen_W.rows(); i++)
				eigen_sW[i] = CMath::sqrt(eigen_W[i]);

			LLT<MatrixXd> L((eigen_sW*eigen_sW.transpose()).cwiseProduct(
				eigen_temp_kernel*m_scale*m_scale) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

			MatrixXd chol = L.matrixL();
			MatrixXd temp2 = L.matrixL();


			MatrixXd temp1 = eigen_sW.rowwise().replicate(eigen_temp_kernel.rows());
		
			MatrixXd arg = eigen_sW.asDiagonal();
			chol = chol.colPivHouseholderQr().solve(arg);

			chol = temp2.transpose().colPivHouseholderQr().solve(chol);

			MatrixXd iKtil = temp1.cwiseProduct(chol);

			VectorXd eigen_alpha = eigen_b - iKtil*(eigen_temp_kernel*eigen_b);

			W = SGVector<float64_t>(eigen_W.rows());

			sW = SGVector<float64_t>(eigen_sW.rows());


			for (index_t i = 0; i < temp2.rows(); i++)
			{
				for (index_t j = 0; j < temp2.cols(); j++)
					m_L(i,j) = temp2(j,i);
			}

			for (index_t i = 0; i < W.vlen; i++)
				W[i] = eigen_W[i];

			for (index_t i = 0; i < sW.vlen; i++)
				sW[i] = eigen_sW[i];

			m_alpha = SGVector<float64_t>(eigen_alpha.rows());

			for (index_t i = 0; i < m_alpha.vlen; i++)
				m_alpha[i] = eigen_alpha[i];

			m_variances = SGVector<float64_t>(variance.rows());

			for (index_t i = 0; i < m_variances.vlen; i++)
				m_variances[i] = variance[i];

			SGVector<float64_t> m_b(eigen_b.rows());
	
			for (index_t i = 0; i < m_b.vlen; i++)
				m_b[i] = eigen_b[i];
		
			m_lz = Psi_New;


}

#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK

