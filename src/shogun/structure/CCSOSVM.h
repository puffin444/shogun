/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2008 Chun-Nam Yu
 */

#ifndef __CCSOSVM_H__
#define __CCSOSVM_H__

#include <shogun/machine/LinearStructuredOutputMachine.h>
#include <shogun/base/DynArray.h>

namespace shogun
{

	/**
	 * Enum
	 * Training method selection
	 */
	enum EQPType
	{
		MOSEK=1,         /**< MOSEK. */
		SVMLIGHT=2       /**< SVM^Light */
	};

	/**
	 *
	 */
	class CCCSOSVM : public CLinearStructuredOutputMachine
	{
		public:
			/** default constructor*/
			CCCSOSVM();

			/** constructor
			 * @param model structured output model
			 * @param w initial w (optional)
			 */
			CCCSOSVM(CStructuredModel* model, SGVector<float64_t> w = SGVector<float64_t>());

			/** destructor */
			virtual ~CCCSOSVM();

			/** @return object name */
			inline virtual const char* get_name() const { return "CCSOSVM"; }

			/** set initial value of weight vector w
			 *
			 * @param W     initial weight vector
			 */
			inline void set_w(SGVector< float64_t > W)
			{
				REQUIRE(W.vlen == m_model->get_dim(), "Dimension of the initial "
						"solution must match the model's dimension!\n");
				m_w=W;
			}

			/** set epsilon
			 *
			 * @param eps epsilon
			 */
			inline void set_epsilon(float64_t eps)
			{
				m_eps = eps;
			}

			/** get epsilon
			 *
			 * @return epsilon
			 */
			inline float64_t get_epsilon() const
			{
				return m_eps;
			}

			/** set C
			 *
			 * @param C constant
			 */
			inline void set_C(float64_t C)
			{
				m_C = C;
			}

			/** get C
			 *
			 * @return C constant
			 */
			inline float64_t get_C() const
			{
				return m_C;
			}

			/** set maximum number of iterations
			 *
			 * @param max_iter maximum number of iterations
			 */
			inline void set_max_iter(index_t max_iter)
			{
				m_max_iter = max_iter;
			}

			/** get maximum number of iterations
			 *
			 * @return maximum number of iterations
			 */
			inline index_t get_max_iter() const
			{
				return m_max_iter;
			}

			/** get the primal objective value
			 *
			 * @return primal objective value
			 */
			inline float64_t compute_primal_objective()
			{
				return m_primal_obj;
			}

			/** get maximum rho value
			 *
			 * @return max rho value
			 */
			inline float64_t get_max_rho() const
			{
				return m_max_rho;
			}

			/** set maximum rho value
			 *
			 * @param max_rho maximum rho value
			 */
			inline void set_max_rho(float64_t max_rho)
			{
				m_max_rho = max_rho;
			}

			/** get the currently used qp solver
			 *
			 * @return qp solver
			 */
			inline EQPType get_qp_type() const
			{
				return m_qp_type;
			}

			/** set the qp solver to be used
			 *
			 * @param type qp solver
			 */
			inline void set_qp_type(EQPType type)
			{
				m_qp_type = type;
			}

		protected:
			bool train_machine(CFeatures* data=NULL);

		private:
			/** find new cutting plane
			 *
			 * @param margin new margin value
			 * @return new cutting plane
			 */
			SGSparseVector<float64_t> find_cutting_plane(float64_t* margin);

			int32_t resize_cleanup(int32_t size_active, SGVector<int32_t>& idle, SGVector<float64_t>&alpha,
					SGVector<float64_t>& delta, SGVector<float64_t>& gammaG0,
					SGVector<float64_t>& proximal_rhs, float64_t ***ptr_G,
					DynArray<SGSparseVector<float64_t> >& dXc, SGVector<float64_t>& cut_error);

			int32_t mosek_qp_optimize(float64_t** G, float64_t* delta, float64_t* alpha, int32_t k, float64_t* dual_obj, float64_t rho);

			/** init class */
			void init();

		private:
			float64_t m_C;
			float64_t m_eps;
			float64_t m_primal_obj;
			float64_t m_alpha_thrld;
			float64_t m_max_rho;

			index_t m_max_iter;
			index_t m_cleanup_check;
			index_t m_idle_iter;

			EQPType m_qp_type;
	};
}

#endif
