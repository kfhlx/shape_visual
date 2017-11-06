/*=========================================================================
  Program: MILX MixView
  Module: itkPCAModelEstimator.txx
  Author: Jurgen Fripp
  Modified by:
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
#ifndef __itkPCAModelEstimator_txx
#define __itkPCAModelEstimator_txx

#include "itkPCAModelEstimator.h"

#include <fstream>
#include <iostream>

#include <itkImportImageFilter.h>

namespace itk
{

template<class TPrecisionType>
PCAModelEstimator<TPrecisionType>
::PCAModelEstimator(void):m_NumberOfTrainingSets(0)
{
  m_EigenVectors.set_size(0,0);
  m_EigenValues.set_size(0);

  m_NumberOfPrincipalComponentsRequired = 1;
  m_NumberOfMeasures = 0;
  m_Valid = false;
}

template<class TPrecisionType>
PCAModelEstimator<TPrecisionType>
::~PCAModelEstimator(void)
{

}

/**
 * PrintSelf
 */
template <class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::PrintSelf( std::ostream& os, Indent indent ) const
{

  os << indent << "                   " << std::endl;
  os << indent << "Models " << std::endl;
  os << indent << "Results printed in the superclass " << std::endl;
  os << indent << "                   " << std::endl;

  Superclass::PrintSelf(os,indent);

  itkDebugMacro(<<"                                    ");
  itkDebugMacro(<<"Results of the model algorithms");
  itkDebugMacro(<<"====================================");

  itkDebugMacro(<< "The eigen values new method are: ");

  itkDebugMacro(<< m_EigenValues);

  itkDebugMacro(<< " ");
  itkDebugMacro(<< "==================   ");

  itkDebugMacro(<< "The eigen vectors new method are: ");


  for(unsigned int i = 0; i < m_EigenValues.size(); i++)
    {
    itkDebugMacro(<< m_EigenVectors.get_row(i));
    }

  itkDebugMacro(<< " ");
  itkDebugMacro(<< "+++++++++++++++++++++++++");

  // Print out ivars
  os << indent << "NumberOfPrincipalComponentsRequired: ";
  os << m_NumberOfPrincipalComponentsRequired << std::endl;
  os << indent << "NumberOfTrainingSets: ";
  os << m_NumberOfTrainingSets << std::endl;


}// end PrintSelf

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::Update()
{
  if(this->GetValid() == false)
    this->GenerateData();
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::AddTrainingSet(VectorType trainingSet)
{
  m_TrainingSets.push_back(trainingSet);
  m_NumberOfTrainingSets = m_TrainingSets.size();
  m_NumberOfMeasures = trainingSet.size();
  this->SetValid(false);
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::GetTrainingSet(int index, vnl_vector<TPrecisionType> &vector)
{
  vector = m_TrainingSets[index];
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::ClearTrainingSets()
{
  int sz = m_TrainingSets.size();
  for (int i = sz-1; i >= 0; i--)
    {
    m_TrainingSets.erase(m_TrainingSets.begin() + i);
    }
  // TODO: Add documentation -> Allows input to be cleared without re-updating built model.
  //this->SetValid(false);
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::GetDecomposition(vnl_vector<TPrecisionType> vector, vnl_vector<TPrecisionType> &decomposition)
{
  this->Update();
  MatrixType modes = this->GetEigenVectors();
  VectorType means = this->GetMeans();

  decomposition = modes.transpose()*(vector - means);
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::GetVectorFromNormalizedDecomposition(vnl_vector<TPrecisionType> decomposition, vnl_vector<TPrecisionType> &reconstruction, int numberOfModes)
{
  // Scale decomposition by eigen-weights
  for(unsigned int i = 0; i < decomposition.size(); i++)
    {
      decomposition(i) *= sqrt(m_EigenValues(i));
    }
  //std::cout << m_EigenValues
  this->GetVectorFromDecomposition(decomposition, reconstruction, numberOfModes);
}


template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::GetVectorFromDecomposition(vnl_vector<TPrecisionType> decomposition, vnl_vector<TPrecisionType> &reconstruction, int numberOfModes)
{
  this->Update();
  if(numberOfModes < 0)
    {
    numberOfModes = decomposition.size();//this->GetNumberOfPrincipalComponentsRequired();
    }
  if(numberOfModes > (int)decomposition.size())
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
      reconstruction(j) += scaling * modes(j,i);
      }
    }
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::GetTrainingMatrix(vnl_matrix<TPrecisionType> &matrix)
{
  matrix.set_size(m_NumberOfTrainingSets, m_NumberOfMeasures);
  for(int i = 0; i < m_NumberOfTrainingSets; i++)
    matrix.set_row(i, m_TrainingSets[i]);
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::SaveTrainingMatrix(const char * filename)
{
  std::ofstream fout(filename, std::ios::binary);

  int sizeData = (int)(m_NumberOfTrainingSets*m_NumberOfMeasures);
  double *writer = new double [sizeData];
  int count = 0;
  for(unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
    {
    for(unsigned int j = 0; j < m_NumberOfMeasures; j++)
      {
      writer[count] = m_TrainingSets[i](j);
      count++;
      }
    }
  fout.write((char *)(writer),sizeData*sizeof(double));
  fout.close();
  delete [] writer;
}

template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::SaveVector(const char * filename, vnl_vector<int> vector)
{
  std::ofstream fout(filename, std::ios::binary);

  int sizeData = vector.size();
  double *writer = new double [sizeData];
  for(int j = 0; j < sizeData; j++)
    {
    writer[j] = vector(j);
    }

  fout.write((char *)(writer),sizeData*sizeof(double));
  fout.close();
  delete [] writer;
}

/**
 * Set the number of required principal components
 */
template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::SetNumberOfPrincipalComponentsRequired( unsigned int n )
{
  if ( m_NumberOfPrincipalComponentsRequired != n )
    {
    m_NumberOfPrincipalComponentsRequired = n;
    this->SetValid(false);
    }
}


template<class TPrecisionType>
template<class TPixel, unsigned Dim>
itk::SmartPointer< itk::Image<TPixel, Dim> >
PCAModelEstimator<TPrecisionType>
::VectorToImage(VectorType &vec, typename itk::Image<TPixel, Dim>::SizeType size, itk::SmartPointer< itk::Image<TPixel, Dim> > image)
{
  ///Convert back to image
  typedef itk::ImportImageFilter<TPixel, Dim> ImportFilterType;
  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();

  typename ImportFilterType::IndexType start;
  start.Fill( 0 );

  typename ImportFilterType::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);

  importFilter->SetRegion( region );
  if(image)
  {
    importFilter->SetOrigin( image->GetOrigin() );
    importFilter->SetSpacing( image->GetSpacing() );
  }

  const bool importImageFilterWillOwnTheBuffer = false;
  importFilter->SetImportPointer( vec.data_block(), vec.size(), importImageFilterWillOwnTheBuffer );
  importFilter->Update();

  return importFilter->GetOutput();
}

/**
 * Generate data (start the model building process)
 */
template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::GenerateData( )
{
  this->EstimateModels();
  this->SetValid(true);
}// end Generate data

/**-----------------------------------------------------------------
 * Takes a set of training sets and returns the means
 * and variance of the various classes defined in the
 * training set.
 */
template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::EstimateModels()
{
  this->EstimatePCAModelParameters();
}// end EstimateShapeModels

/**-----------------------------------------------------------------
 *Estimage shape models using PCA.
 *-----------------------------------------------------------------
 */
template<class TPrecisionType>
void
PCAModelEstimator<TPrecisionType>
::EstimatePCAModelParameters()
{
  //-------------------------------------------------------------------------
  //Calculate the Means
  //-------------------------------------------------------------------------
  //std::cout << "PCAModelEstimator: Make Mean " << m_NumberOfMeasures << std::endl;
  m_Means.set_size(m_NumberOfMeasures);
  m_Means.fill(0);

  for(unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
    {
    m_Means += m_TrainingSets[i];
    }
  m_Means /= (TPrecisionType)(m_NumberOfTrainingSets);
  //std::cout << "PCAModelEstimator: Mean Performed " << m_Means.size() << " " << m_NumberOfTrainingSets << std::endl;

  //std::cout << "PCAModelEstimator: Make D" << std::endl;
  vnl_matrix<TPrecisionType> D;
  D.set_size(m_NumberOfMeasures, m_NumberOfTrainingSets);
  D.fill(0);

  for(unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
    {
    D.set_column(i, m_TrainingSets[i] - m_Means);
    }
  //std::cout << "PCAModelEstimator: D Performed " << D.rows() << " " << D.columns() << std::endl;

  vnl_matrix<TPrecisionType> T = (D.transpose()*D)/(m_NumberOfTrainingSets-1);

  //std::cout << "PCAModelEstimator: T Performed " << T.rows() << " " << T.columns() << std::endl;

  m_EigenValues.set_size(m_NumberOfTrainingSets);
  m_EigenVectors.set_size(m_NumberOfTrainingSets,m_NumberOfTrainingSets);

  //std::cout << "PCAModelEstimator: Solving Eigensystem" << std::endl;

  vnl_symmetric_eigensystem_compute(T, m_EigenVectors, m_EigenValues);

  //Flip the eigen values since the eigen vectors output
  //is ordered in decending order of their corresponding eigen values.
  m_EigenValues.flip();
  m_EigenVectors.fliplr();
  //std::cout << "PCAModelEstimator: Eigensystem2" << std::endl;
  m_EigenVectors = D*m_EigenVectors;
  //std::cout << "PCAModelEstimator: Eigensystem3" << m_EigenVectors << std::endl;
  m_EigenVectors.normalize_columns();
  //std::cout << "PCAModelEstimator: Eigensystem4 " << m_EigenVectors << std::endl;

  for(unsigned int i = 0; i < m_EigenValues.size(); i++)
    {
    if(m_EigenValues(i) < 0)
      {
      itkDebugMacro(<< "Eigenvalue " << i << " " << m_EigenValues(i) << " set to 0 ");
      m_EigenValues(i) = 0;
      }
    }

  //std::cout << m_EigenVectors.size() << std::endl;
  //std::cout << D.rows() << " " << D.columns() << std::endl;
  //std::cout << m_EigenVectors.rows() << " " << m_EigenVectors.columns() << std::endl;

}// end EstimatePCAModelParameters

//-----------------------------------------------------------------


} // namespace itk

#endif
