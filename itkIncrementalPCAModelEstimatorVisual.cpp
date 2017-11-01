#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
//ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImportImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkMesh.h>
#include "itkVTKPolyDataReader.h"

//VTK
#include <vtkFloatArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPointData.h>
//VNL
#include <vnl/vnl_matrix_ref.h>
//SMILX
#include <milxImage.h>

#include "itkIncrementalPCAModelEstimator.h"

const unsigned Dimension = 3;
typedef float InputPixelType;
typedef float OutputPixelType;
typedef InputPixelType PrecisionType;
typedef itk::Image<InputPixelType, Dimension> InputImageType;
typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
typedef InputImageType::SizeType SizeType;
typedef itk::IncrementalPCAModelEstimator<PrecisionType> IncrementalPCAModelEstimatorType;
typedef vnl_matrix<PrecisionType> MatrixType;
typedef vnl_vector<PrecisionType> VectorType;
bool theEnd = false;

bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames);

int main(int argc, char * argv[])
{
	if (argc < 2)
	{
		std::cerr << "Shape Modelling App" << std::endl;
		std::cerr << "Assumes meshes in MVB are Polydata." << std::endl;
		std::cerr << "Usage:" << std::endl;
		std::cerr << "mvb file" << std::endl;
		//std::cerr << "BatchPCA Size" << std::endl;
		//std::cerr << "Eigenvalue Size Control" << std::endl;
		std::cerr << "trainingSets Size Control" << std::endl;
		//std::cerr << "mode" << std::endl;
		//std::cerr << "weight" << std::endl;
		//std::cerr << "precision: 0-1" << std::endl;
		return EXIT_FAILURE;
	}
	std::string inputFileName = argv[1];
	//int batchSize = atoi(argv[2]);
	//int eigenvalueSizeControl = atof(argv[3]);
	//int trainingSetsSizeControl = atoi(argv[4]);
	//int mode = atoi(argv[5]);
	//double weight = atof(argv[6]);
	//double precision = atof(argv[7]);
	int trainingSetsSizeControl = atoi(argv[2]);

	std::vector<int> ids;
	std::vector<std::string> filenames;
	std::string fileExtension("mvb");

	IncrementalPCAModelEstimatorType::Pointer ipcaModel = IncrementalPCAModelEstimatorType::New();

	if (inputFileName.find(fileExtension) != std::string::npos)
	{
		std::cout << "MVB Extension found " << inputFileName << std::endl;
		bool success = ReadSurfaceFileNames(inputFileName.c_str(), ids, filenames);
		if (success == EXIT_FAILURE)
		{
			std::cout << "Failed to read " << inputFileName.c_str() << std::endl;
			return EXIT_FAILURE;
		}
	}
	else
	{
		std::cout << "InputFile Not an MVB file " << inputFileName.c_str() << std::endl;
		return EXIT_FAILURE;
	}

	typedef itk::Mesh<PrecisionType, 3> MeshType;
	typedef itk::VTKPolyDataReader< MeshType > ReaderType;
	typedef ReaderType::PointType PointType;

	int count = 1;
	int numberOfPoints;
	for (int i = 0; i < trainingSetsSizeControl; i++)
	{
		ReaderType::Pointer  polyDataReader = ReaderType::New();
		polyDataReader->SetFileName(filenames[i].c_str());
		std::cout << "Adding ID " << count << " " << filenames[i].c_str() << std::endl;
		count++;
		try
		{
			polyDataReader->Update();
		}
		catch (itk::ExceptionObject & excp)
		{
			std::cerr << "Error during Update() " << std::endl;
			std::cerr << excp << std::endl;
			return EXIT_FAILURE;
		}

		MeshType::Pointer mesh = polyDataReader->GetOutput();
		numberOfPoints = mesh->GetNumberOfPoints();
		unsigned int numberOfCells = mesh->GetNumberOfCells();

		// Retrieve points
		VectorType pointsVector(3 * numberOfPoints); //for each x, y, z values
		for (int i = 0; i < numberOfPoints; i++)
		{
			PointType pp;
			bool pointExists = mesh->GetPoint(i, &pp);
			if (pointExists)
			{
				//std::cout << "Point is = " << pp << std::endl;
				pointsVector[(i * 3)] = pp[0];
				pointsVector[(i * 3) + 1] = pp[1];
				pointsVector[(i * 3) + 2] = pp[2];
			}
		}
		ipcaModel->AddTrainingSet(pointsVector);
	}
	while (theEnd == false) // while loop reduce adding training set again
	{
		int batchSize;
		double precision;
		int eigenvalueSizeControl;
		double weight;
		int mode;

		std::cout << "input batchSize, precision, eigenvalueSizeControl, mode, weight: " << std::endl;
		std::cin >> batchSize >> precision >> eigenvalueSizeControl >> mode >> weight;
		std::cout << "batchSize: " << batchSize << std::endl;
		std::cout << "precision: " << precision << std::endl;
		std::cout << "eigenvalueSizeControl: " << eigenvalueSizeControl << std::endl;
		std::cout << "mode: " << mode << std::endl;
		std::cout << "weight: " << weight << std::endl;

		ipcaModel->setPCABatchSize(batchSize);
		ipcaModel->setPrecision(precision);
		ipcaModel->seteigenvalueSize(eigenvalueSizeControl);

		//Timing
		std::clock_t start;
		double duration;
		start = std::clock();

		ipcaModel->Update();

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "time: " << duration << '\n';

		VectorType eigenValues = ipcaModel->GetEigenValues();
		unsigned int numEigVal = eigenValues.size();
		std::cout << "Number of returned eign-values: " << numEigVal << std::endl;
		for (unsigned int i = 0; i< numEigVal; i++)
		{
			std::cout << eigenValues[i] << ", ";
		}
		std::cout << std::endl;
		/**
		* write eigenValues to file
		*/
		std::ofstream myfile;
		myfile.open("C:\\Users\\Alex\\Desktop\\eigenvalue.csv");
		myfile << "Mode,Value,\n";
		for (int i = 0; i < eigenValues.size(); i++)
		{
			myfile << i + 1 << "," << eigenValues.get(i) << "\n";
		}
		myfile.close();



		/**
		* visualisation
		*/
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
		vtkSmartPointer<vtkPoints> meanpoints = vtkSmartPointer<vtkPoints>::New();

		VectorType b(mode, 0);
		b(mode - 1) = weight; //!< View mode
		cout << "Generating display for mode " << mode << ", weight " << weight << endl;
		VectorType recon;
		ipcaModel->GetVectorFromDecomposition(b, recon);
		std::cout << "recon: " << recon.size() << std::endl;

		std::ofstream myfile2;
		myfile2.open("C:\\Users\\Alex\\Desktop\\recon.csv");
		myfile2 << "recon,\n";
		for (int i = 0; i < recon.size(); i++)
		{
			myfile2 << i + 1 << "," << recon.get(i) << "\n";
		}
		myfile2.close();

		////RMSE
		//float rmse = 0.0;
		//for (unsigned j = 0; j < leftOutImageVector.size(); j++)
		//{
		//	float value = leftOutImageVector[j] - reconstruction[j];
		//	rmse += value*value;
		//}
		//rmse /= leftOutImageVector.size();
		//std::cerr << "RMSE: " << sqrt(rmse) << std::endl;

		VectorType means;
		means = ipcaModel->GetMeans(); // repeat the 
		for (int i = 0; i < numberOfPoints; i++)
		{
			PrecisionType x = means(i * 3);
			PrecisionType y = means((i * 3) + 1);
			PrecisionType z = means((i * 3) + 2);
			meanpoints->InsertNextPoint(x, y, z);
		}

		// from vector to polydata

		for (int i = 0; i < numberOfPoints; i++)
		{
			PrecisionType x = recon(i * 3);
			PrecisionType y = recon((i * 3) + 1);
			PrecisionType z = recon((i * 3) + 2);
			points->InsertNextPoint(x, y, z);
		}

		vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
		reader->SetFileName(filenames[0].c_str());
		reader->Update();

		vtkSmartPointer<vtkPolyData> meanpolyData = reader->GetOutput();
		meanpolyData->SetPoints(meanpoints);
		vtkSmartPointer<vtkPolyData> meanShape = vtkSmartPointer<vtkPolyData>::New();
		meanShape->DeepCopy(meanpolyData); //use GetVectorFromDecomposition()

		vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();
		polyData->SetPoints(points);
		vtkSmartPointer<vtkPolyData> varShape = vtkSmartPointer<vtkPolyData>::New();
		varShape->DeepCopy(polyData); //use GetVectorFromDecomposition()

		vtkSmartPointer<vtkFloatArray> varScalars = vtkSmartPointer<vtkFloatArray>::New();
		varScalars->SetName("Variation");
		vtkSmartPointer<vtkFloatArray> varVectors = vtkSmartPointer<vtkFloatArray>::New();
		varVectors->SetNumberOfComponents(3);
		vtkSmartPointer<vtkFloatArray> varShapeVectors = vtkSmartPointer<vtkFloatArray>::New();
		varShapeVectors->SetNumberOfComponents(3);

		//std::cout << "Computing Variation" << std::endl;
		for (int i = 0; i < meanShape->GetNumberOfPoints(); i++)
		{
			double* meanPoint = meanShape->GetPoint(i); // meanshape is m_Means
			double* varPoint = varShape->GetPoint(i);

			double xVal = varPoint[0] - meanPoint[0];
			double yVal = varPoint[1] - meanPoint[1];
			double zVal = varPoint[2] - meanPoint[2];

			double var;
			var = xVal*xVal + yVal*yVal + zVal*zVal;

			// std::cout << var << std::endl;
			varVectors->InsertNextTuple3(xVal, yVal, zVal);
			varShapeVectors->InsertNextTuple3(-xVal, -yVal, -zVal);
			varScalars->InsertNextValue(var);
		}

		vtkFloatingPointType maxScalar = std::numeric_limits<vtkFloatingPointType>::min();
		for (int i = 0; i < varScalars->GetNumberOfTuples(); i++)
		{
			if (varScalars->GetValue(i) > maxScalar)
				maxScalar = varScalars->GetValue(i);
		}

		for (int i = 0; i < varScalars->GetNumberOfTuples(); i++)
		{
			vtkFloatingPointType var = varScalars->GetValue(i);
			varScalars->SetValue(i, var / maxScalar);
		}
		// save to file
		// try vtk 6/7
		meanShape->GetPointData()->SetVectors(varVectors);
		varShape->GetPointData()->SetVectors(varShapeVectors);
		meanShape->GetPointData()->SetScalars(varScalars);
		// Write the file
		vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer->SetFileName("C:\\Users\\Alex\\Desktop\\meanShape.vtp");
		writer->SetInputData(meanShape);
		writer->Write();
		vtkSmartPointer<vtkXMLPolyDataWriter> writer2 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer2->SetFileName("C:\\Users\\Alex\\Desktop\\varShape.vtp");
		writer2->SetInputData(varShape);
		writer2->Write();

		std::cout << "file write complete." << std::endl;
		char flag;
		std::cout << "continue? (Y/N) ";
		std::cin >> flag;
		if (flag == 'Y')
			theEnd = false;
		if (flag == 'N')
			theEnd = true;
	}
	//Add to PCA model
	
}

bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames)
{
	std::fstream inFile(filename, std::ios::in);
	if (inFile.fail())
	{
		std::cerr << "Cannot read input data file " << filename << std::endl;
		return false;
	}

	std::string bufferKey;
	inFile >> bufferKey;
	std::string szKey = "MILXVIEW_BATCH_FILE";
	if (szKey != bufferKey)
	{
		std::cerr << "Invalid input data file " << filename << std::endl;
		std::cerr << bufferKey << std::endl;
		std::cerr << szKey << std::endl;
		return false;
	}

	std::string key, name;
	std::string descKey = "CASE_SURFACE";
	std::string descKey2 = "CASE_IMAGE";
	int id;
	while (inFile >> key >> id >> name)
	{
		if (key == descKey || key == descKey2)
		{
			ids.push_back(id);
			filenames.push_back(name);
			//std::cout << id << " " << name << " " << std::endl;
		}
		else
		{
			std::cerr << "Seems to have incorrect data" << std::endl;
			return false;
		}
	}
	inFile.close();
	std::cout << "ReadImageFileNames Finished" << std::endl;

	return EXIT_SUCCESS;
}

//C:\Users\Alex\Documents\shape_visual\build\bin\Release\itkIncrementalPCAModelEstimatorVisual.exe C:\Users\Alex\Documents\thesis\IncrementalLearn\IPCA\aligned\aligned.mvb 100