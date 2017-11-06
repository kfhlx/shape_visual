/*=========================================================================
  Program: MILX MixView
  Module: itkIncrementalPCAModelEstimator.txx
  Author: Jurgen Fripp
  Modified by: King Fai Ho
  Language: C++
  Created: Fri 09 March 2007 16:21:00 EST

  Copyright: (c) 2009 CSIRO, Australia.

  This software is protected by international copyright laws.
  Any unauthorised copying, distribution or reverse engineering is prohibited.

  Licence:
  All rights in this Software are reserved to CSIRO. You are only permitted
  to have this Software in your possession and to make use of it if you have
  agreed to a Software License with CSIRO.

  BioMedIA Lab: http://www.ict.csiro.au/BioMedIA/
=========================================================================*/
#ifndef __itkIncrementalPCAModelEstimator_txx
#define __itkIncrementalPCAModelEstimator_txx

#include "itkIncrementalPCAModelEstimator.h"

namespace itk
{
	template<class TPrecisionType>
	IncrementalPCAModelEstimator<TPrecisionType>
		::IncrementalPCAModelEstimator(void) :m_NumberOfTrainingSets(0)
	{
		m_EigenVectors.set_size(0, 0);
		m_EigenValues.set_size(0);
		m_A = 0;
		m_NumberOfPrincipalComponentsRequired = 1;
		m_NumberOfMeasures = 0;
		m_Valid = false;
	}

	template<class TPrecisionType>
	IncrementalPCAModelEstimator<TPrecisionType>
		::~IncrementalPCAModelEstimator(void)
	{

	}

	/**
	 * PrintSelf
	 */
	template <class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf(os, indent);
	}// end PrintSelf

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::Update()
	{
		if (this->GetValid() == false)
			this->GenerateData();
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::AddTrainingSet(VectorType trainingSet)
	{
		m_TrainingSets.push_back(trainingSet);
		m_NumberOfTrainingSets = m_TrainingSets.size();
		m_NumberOfMeasures = trainingSet.size();
		this->SetValid(false);
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::setPCABatchSize(int batchSize)
	{
		m_batchSize = batchSize;
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::setPrecision(double precision)
	{
		m_Precision = precision;
	}

	template<class TPrecisionType>
	void IncrementalPCAModelEstimator<TPrecisionType>::setEigenvalueSizeControl(int eigenvalueSizeControl)
	{
		m_eigenvalueSizeControl = eigenvalueSizeControl;
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::GetTrainingSet(int index, vnl_vector<TPrecisionType> &vector)
	{
		vector = m_TrainingSets[index];
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::ClearTrainingSets()
	{
		int sz = m_TrainingSets.size();
		for (int i = sz - 1; i >= 0; i--)
		{
			m_TrainingSets.erase(m_TrainingSets.begin() + i);
		}
		// TODO: Add documentation -> Allows input to be cleared without re-updating built model.
		//this->SetValid(false);
	}
	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::GetDecomposition(VectorType vector, VectorType &decomposition)
	{
		Superclass::GetDecomposition(vector, decomposition);
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::GetVectorFromNormalizedDecomposition(vnl_vector<TPrecisionType> decomposition, vnl_vector<TPrecisionType> &reconstruction, int numberOfModes)
	{
		// Scale decomposition by eigen-weights
		for (unsigned int i = 0; i < decomposition.size(); i++)
		{
			decomposition(i) *= sqrt(m_EigenValues(i));
		}
		//std::cout << m_EigenValues
		this->GetVectorFromDecomposition(decomposition, reconstruction, numberOfModes);
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::GetVectorFromDecomposition(vnl_vector<TPrecisionType> decomposition, vnl_vector<TPrecisionType> &reconstruction, int numberOfModes)
	{
		//this->Update();
		if (numberOfModes < 0)
		{
			numberOfModes = decomposition.size();//this->GetNumberOfPrincipalComponentsRequired();
		}
		if (numberOfModes > (int)decomposition.size())
		{
			numberOfModes = decomposition.size();
			std::cout << "Using " << numberOfModes << std::endl;
		}

		reconstruction = this->GetMeans();
		VectorType eigenValues = this->GetEigenValues();
		MatrixType modes = this->GetEigenVectors();

		int size = reconstruction.size();

		for (int i = 0; i < numberOfModes; i++)
		{
			TPrecisionType scaling = decomposition(i);
			for (int j = 0; j < size; j++)
			{
				reconstruction(j) += scaling * modes(j, i);
			}
		}
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::GetTrainingMatrix(vnl_matrix<TPrecisionType> &matrix)
	{
		matrix.set_size(m_NumberOfTrainingSets, m_NumberOfMeasures);
		for (int i = 0; i < m_NumberOfTrainingSets; i++)
			matrix.set_row(i, m_TrainingSets[i]);
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::SaveTrainingMatrix(const char * filename)
	{
		std::ofstream fout(filename, std::ios::binary);

		int sizeData = (int)(m_NumberOfTrainingSets*m_NumberOfMeasures);
		double *writer = new double[sizeData];
		int count = 0;
		for (unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
		{
			for (unsigned int j = 0; j < m_NumberOfMeasures; j++)
			{
				writer[count] = m_TrainingSets[i](j);
				count++;
			}
		}
		fout.write((char *)(writer), sizeData * sizeof(double));
		fout.close();
		delete[] writer;
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::SaveVector(const char * filename, vnl_vector<int> vector)
	{
		std::ofstream fout(filename, std::ios::binary);

		int sizeData = vector.size();
		double *writer = new double[sizeData];
		for (int j = 0; j < sizeData; j++)
		{
			writer[j] = vector(j);
		}

		fout.write((char *)(writer), sizeData * sizeof(double));
		fout.close();
		delete[] writer;
	}

	/**
	 * Generate data (start the model building process)
	 */
	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::GenerateData()
	{
		this->EstimateModels();
		//this->SetValid(true);
	}// end Generate data

	/**-----------------------------------------------------------------
	 * Takes a set of training sets and returns the means
	 * and variance of the various classes defined in the
	 * training set.
	 */
	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::EstimateModels()
	{
		this->IPCAModelParameters();

	}// end EstimateShapeModels

	/**-----------------------------------------------------------------
	 *Estimage shape models using PCA and iPCA.
	 *-----------------------------------------------------------------
	 */
	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::EstimatePCAModelParameters()
	{
		// calculate mean
		m_Means.set_size(m_NumberOfMeasures);
		m_Means.fill(0);
		for (int i = 0; i < m_batchSize; i++)
		{
			m_Means += m_TrainingSets[i];
		}
		m_Means /= (PrecisionType)(m_NumberOfTrainingSets);

		// construct D
		MatrixType D;
		D.set_size(m_NumberOfMeasures, m_batchSize);
		D.fill(0);
		// remove mean
		for (int i = 0; i < m_batchSize; i++)
		{
			const VectorType tmpSet = m_TrainingSets[i] - m_Means;
			D.set_column(i, tmpSet);
		}
		m_EigenValues.set_size(m_batchSize);
		m_EigenVectors.set_size(m_NumberOfMeasures, m_batchSize);

		ApplyStandardPCA(D, m_EigenVectors, m_EigenValues);

		// coefficient m_A
		m_A.set_size(m_batchSize, m_batchSize);
		for (int i = 0; i < m_batchSize; i++)
		{
			m_A.set_column(i, m_EigenVectors.transpose() * D.get_column(i));
		}
		//	GetTrainingMatrix(m_D);
		//	const VectorType onesCol(this->m_NumberOfTrainingSets, 1.0);
		//	const MatrixType MeansMatrix = outer_product(this->m_Means, onesCol);
		//	TPrecisionType error = Reconstruct(recon, m_D, m_EigenVectors, m_A, MeansMatrix);


	}// end EstimatePCAModelParameters

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::IPCAModelParameters()
	{
		this->EstimatePCAModelParameters();
		cout << "EstimatePCAModelParameters() done" << endl;

		/* variables for ipca */
		MatrixType UT, Ud, Udd, Ad, oldEVec, m_D, recon;
		VectorType x, a, y, r, rn, lamdadd, udd, dummy;
		bool startTrim = false;
		bool trimtrigger = false;

		/*GetTrainingMatrix(m_D);
		const VectorType onesCol(this->m_NumberOfTrainingSets, 1.0);
		const MatrixType MeansMatrix = outer_product(this->m_Means, onesCol);
		TPrecisionType error = Reconstruct(recon, m_D, m_EigenVectors, m_A, MeansMatrix);*/
		//if (m_batchSize == 0) // complete incremental, start with IPCA
		//{
		//	cout << "full incremental" << endl;
		//	m_EigenVectors.set_size(GetNumberOfMeasures(), 1);
		//	m_EigenVectors.fill(0);
		//	m_A = 0;
		//	m_Means = m_TrainingSets[0];
		//	m_batchSize++;
		//}


		
		for (unsigned int i = m_batchSize; i < m_NumberOfTrainingSets; i++)
		{
			// 1. Project new surface from D to current eigenspace, a = UT(x-mean)
			UT = m_EigenVectors.transpose();
			x = m_TrainingSets[i]; // new image
			a = UT * (x - m_Means);

			// 2. Reconstruct new image, y = U a + mean
			y = m_EigenVectors * a + m_Means;

			// 3. Compute the residual vector, r is orthogonal to U
			r = x - y;

			// 4. Append r as a new basis vector
			Ud.set_size(m_NumberOfMeasures, m_EigenVectors.cols() + 1);
			Ud.set_columns(0, m_EigenVectors);
			Ud.set_column(Ud.cols() - 1, r / r.two_norm());

			// 5. New coefficients 
			Ad.set_size(m_A.rows() + 1, m_A.rows() + 1); // i+1 x i+1
			Ad.fill(0);
			Ad.update(m_A, 0, 0); // add A at top left corner
			Ad.set_column(Ad.cols() - 1, a); // add a at last column
			Ad.put(Ad.rows() - 1, Ad.cols() - 1, r.two_norm()); // add ||r|| at bottom right corner
			//cout << "r.two_norm(): " << r.two_norm() << endl;

			// 6. Perform PCA on Ad, obtain udd, Udd, lamdadd
			ApplyStandardPCA2(Ad, Udd, lamdadd);
			// trim trigger
			if (trimtrigger)
			{

			}

			// 7. Project the coefficient vectors to new basis 
			udd.set_size(Ad.cols());
			udd.fill(0);
			for (unsigned int j = 0; j < Ad.cols(); j++)
			{
				udd += Ad.get_column(j);
			}
			udd /= (PrecisionType)(Ad.cols());
			for (unsigned int j = 0; j < Ad.cols(); j++) // remove udd(mean of Ad) from Ad
			{
				const VectorType tmpSet = Ad.get_column(j) - udd;
				Ad.set_column(j, tmpSet);
			}
			m_A = Udd.transpose() * Ad;

			// 8. Rotate the subspace
			m_EigenVectors = Ud * Udd;

			// 9. Update the mean
			m_Means = m_Means + Ud * udd;

			// 10. New eigenvalues
			m_EigenValues = lamdadd;

			if (m_EigenValues.size() > 20)
			{
				trimtrigger = true;
			}
			cout << i << "/" << m_NumberOfTrainingSets << "\r" ;

			// reduce model size
			//if (m_EigenValues.size() > 20)
			//{
			//	startTrim = true;
			//}
			//if (startTrim == true)
			//{
			//	cout << "start trim" << endl;
			//	int model_index = GetNumberOfModesRequired(m_Precision);
			//	//std::cout << "model_index: " << model_index << std::endl;
			//	// discard size of eigenvector and eigenvalue after the model index
			//	// eigenvector
			//	MatrixType tmpEvec;
			//	tmpEvec.set_size(m_NumberOfMeasures, model_index);
			//	tmpEvec = m_EigenVectors.extract(m_NumberOfMeasures, model_index, 0, 0); //??
			//	oldEVec = m_EigenVectors;
			//	m_EigenVectors = tmpEvec;
			//	//std::cout << "m_EigenVectors: " << m_EigenVectors.rows() << " " << m_EigenVectors.cols() << std::endl;
			//	// eigenvalue
			//	VectorType tmpEval;
			//	tmpEval.set_size(model_index);
			//	tmpEval = m_EigenValues.extract(model_index, 0);
			//	m_EigenValues = tmpEval;
			//	//std::cout << "m_EigenValues: " << m_EigenValues.size() << std::endl;
			//	// m_A
			//	// trimming the coefficient
			//	// need to fix
			//	// #1
			//	VectorType dummy;
			//	MatrixType recon = oldEVec*m_A;
			//	ApplyStandardPCA(recon, oldEVec, dummy);
			//	m_A.set_size(model_index, model_index);
			//	for (int j = 0; j < model_index; j++)
			//	{
			//		m_A.set_column(j, oldEVec.transpose() * recon.get_column(j));
			//	}
			//	// #2
			//	startTrim = false;
			//}	
		}// ipca for loop end
		//startTrim = false;

		if (m_batchSize != m_NumberOfTrainingSets) // reconstruction
		{
			GetTrainingMatrix(m_D);
			const VectorType onesCol(this->m_NumberOfTrainingSets, 1.0);
			const MatrixType MeansMatrix = outer_product(this->m_Means, onesCol);
			TPrecisionType error = Reconstruct(recon, m_D, m_EigenVectors, m_A, MeansMatrix);
			VectorType dummy;
			oldEVec = m_EigenVectors;
			MatrixType recon2 = oldEVec*m_A;
			ApplyStandardPCA(recon2, m_EigenVectors, dummy); // revert the m_eigenvector for vtk recon visual
		}
		
		//// trim eigenvectorSize if needed
		//if (m_EigenValues.size() > m_eigenvalueSizeControl)
		//{
		//	m_EigenValues = m_EigenValues.extract(m_eigenvalueSizeControl, 0);
		//}
	}

	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::ApplyStandardPCA(const MatrixType &data, MatrixType &eigenVecs, VectorType &eigenVals)
	{
		// covariance
		const PrecisionType norm = 1.0 / (data.cols() - 1);
		const vnl_matrix<PrecisionType> T = (data.transpose()*data)*norm; //D^T.D is smaller so more efficient
		//SVD
		vnl_svd<PrecisionType> svd(T); //!< Form Projected Covariance matrix and compute SVD, ZZ^T
		svd.zero_out_absolute(); ///Zero out values below 1e-8 but greater than zero
		///pinverse unnecessary?
		//eigenVecs = data*vnl_matrix_inverse<double>(svd.U()).pinverse().transpose(); //!< Extract eigenvectors from U, noting U = V^T since covariance matrix is real and symmetric
		eigenVecs = data*svd.U(); //!< Extract eigenvectors from U, noting U = V^T since covariance matrix is real and symmetric
		eigenVecs.normalize_columns();
		eigenVals = svd.W().diagonal();
	}
	template<class TPrecisionType>
	void
		IncrementalPCAModelEstimator<TPrecisionType>
		::ApplyStandardPCA2(const MatrixType &data, MatrixType &eigenVecs, VectorType &eigenVals)
	{
		// covariance
		const PrecisionType norm = 1.0 / (data.cols() - 1);
		const vnl_matrix<PrecisionType> T = (data.transpose()*data)*norm; //D^T.D is smaller so more efficient
		//SVD
		vnl_svd<PrecisionType> svd(T); //!< Form Projected Covariance matrix and compute SVD, ZZ^T
		///svd.zero_out_absolute(); ///Zero out values below 1e-8 but greater than zero
		eigenVecs = svd.U();
		eigenVecs.normalize_columns();
		eigenVals = svd.W().diagonal();
	}
	template<class TPrecisionType>
	TPrecisionType
		IncrementalPCAModelEstimator<TPrecisionType>
		::Reconstruct(MatrixType &recon, const MatrixType &data, const MatrixType &eigenVecs, const MatrixType &coefficients, const MatrixType &means)
	{
		int numberOfModes = this->GetNumberOfModesRequired(m_Precision);
		std::cerr << "Modes in Reconstruction: " << numberOfModes << std::endl;
		MatrixType truncatedVecs(this->m_NumberOfMeasures, this->m_NumberOfTrainingSets, 0.0);
		for (int j = 0; j < numberOfModes; j++)
			truncatedVecs.set_column(j, eigenVecs.get_column(j));
		recon = truncatedVecs*coefficients + means; ///Reconstruct all shapes
													//  recon = eigenVecs*coefficients + means; ///Reconstruct all shapes
													///Compute difference of steps for convergence testing
		const MatrixType Error = data - recon; ///Reconstruction error
		TPrecisionType error = Error.rms();
		std::cerr << "Reconstruction MSE of " << error << std::endl;

		return error;
	}
	//-----------------------------------------------------------------
} // namespace itk

#endif