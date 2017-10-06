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
//VNL
#include <vnl/vnl_matrix_ref.h>
//SMILIy
#include <C:\Program Files\SMILX\include\milxImage.h>

#include "itkIncrementalPCAModelEstimator.h"

const unsigned Dimension = 3;
typedef float InputPixelType;
typedef float OutputPixelType;
typedef InputPixelType PrecisionType;
typedef itk::Image<InputPixelType, Dimension> InputImageType;
typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
typedef InputImageType::SizeType SizeType;
typedef itk::IncrementalPCAModelEstimator<PrecisionType> IncrementalPCAModelEstimatorType;
typedef vnl_matrix<PrecisionType>  MatrixType;
typedef vnl_vector<PrecisionType> VectorType;

bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames);

int main(int argc, char * argv[])
{
	if (argc < 4)
	{
		std::cerr << "Shape Modelling App" << std::endl;
		std::cerr << "Assumes meshes in MVB are Polydata." << std::endl;
		std::cerr << "Usage:" << std::endl;
		std::cerr << "mvb file" << std::endl;
		std::cerr << "BatchPCA Size " << std::endl;
		std::cerr << "Eigenvector Size" << std::endl;
		std::cerr << "trainingSets Size control" << std::endl;
		//std::cerr << "weight" << std::endl;
		//std::cerr << "mode" << std::endl;
		return EXIT_FAILURE;
	}
	std::string inputFileName = argv[1];
	int batchSize = atoi(argv[2]);
	int eigenvectorSize = atof(argv[3]);
	int trainingSetsSizeControl = atoi(argv[4]);
	//int weight = atoi(argv[5]);
	//int mode = atoi(argv[6]);

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
	typedef vnl_vector<PrecisionType> VectorType;
	typedef vnl_matrix<PrecisionType> MatrixType;

	int count = 1;
	unsigned int numberOfPoints;
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
		//std::cout << "numberOfPoints= " << numberOfPoints << std::endl;
		//std::cout << "numberOfCells= " << numberOfCells << std::endl;

		// Retrieve points
		VectorType pointsVector(3 * numberOfPoints); //for each x, y, z values
		for (unsigned int i = 0; i < numberOfPoints; i++)
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
	//Add to PCA model
	ipcaModel->setPCABatchSize(batchSize);
	ipcaModel->Update();

	vnl_vector<PrecisionType> eigenValues = ipcaModel->GetEigenValues();
	unsigned int numEigVal = eigenValues.size();
	std::cout << "Number of returned eign-values: " << numEigVal << std::endl;

	for (unsigned int i = 0; i< numEigVal; i++)
	{
		std::cout << eigenValues[i] << ", ";
	}
	std::cout << std::endl;

	/*insert visualisation*/
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	int mode = 1;
	double weight = 3;
	vnl_vector<PrecisionType> b(mode, 0);
	b(mode - 1) = weight; //!< View mode
	cout << "Generating display for mode " << mode - 1 << endl;
	vnl_vector<PrecisionType> recon;
	ipcaModel->GetVectorFromDecomposition(b, recon);

	// from vector to polydata
	// put into a function to also get the mean shape

	for (int i = 0; i < numberOfPoints; i++)
	{
		PrecisionType x = recon(i);
		PrecisionType y = recon(i + 1);
		PrecisionType z = recon(i + 2);
		points->InsertNextPoint(x, y, z);
	}

	vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
	reader->SetFileName(filenames[0].c_str());
	reader->Update();

	vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();
	polyData->SetPoints(points);
	vtkSmartPointer<vtkPolyData> varShape = vtkSmartPointer<vtkPolyData>::New();
	varShape->DeepCopy(polyData); //use GetVectorFromDecomposition()

								  //rest goes here
	vtkSmartPointer<vtkFloatArray> varScalars = vtkSmartPointer<vtkFloatArray>::New();
	varScalars->SetName("Variation");
	vtkSmartPointer<vtkFloatArray> varVectors = vtkSmartPointer<vtkFloatArray>::New();
	varVectors->SetNumberOfComponents(3);
	vtkSmartPointer<vtkFloatArray> varShapeVectors = vtkSmartPointer<vtkFloatArray>::New();
	varShapeVectors->SetNumberOfComponents(3);
	vtkSmartPointer<vtkFloatArray> varTensors = vtkSmartPointer<vtkFloatArray>::New();
	varTensors->SetNumberOfComponents(9);

	//printInfo("Computing Variation");
	//for (int i = 0; i < meanShape->GetNumberOfPoints(); i++)
	//{
	//	vtkFloatingPointType* meanPoint = meanShape->GetPoint(i);
	//	vtkFloatingPointType* varPoint = varShape->GetPoint(i);

	//	vtkFloatingPointType xVal = varPoint[0] - meanPoint[0];
	//	vtkFloatingPointType yVal = varPoint[1] - meanPoint[1];
	//	vtkFloatingPointType zVal = varPoint[2] - meanPoint[2];

	//	float normalPoint[3];
	//	normArray->GetTupleValue(i, normalPoint);
	//	vtkFloatingPointType var;
	//	if (flgNormal)
	//	{
	//		var = xVal*normalPoint[0] + yVal*normalPoint[1] + zVal*normalPoint[2];
	//		var = var*var;
	//	}
	//	else
	//	{
	//		var = xVal*xVal + yVal*yVal + zVal*zVal;
	//	}
	//	// std::cout << var << std::endl;
	//	varVectors->InsertNextTuple3(xVal, yVal, zVal);
	//	varShapeVectors->InsertNextTuple3(-xVal, -yVal, -zVal);
	//	varScalars->InsertNextValue(var);
	//}

	//vtkFloatingPointType maxScalar = std::numeric_limits<vtkFloatingPointType>::min();
	//for (int i = 0; i < varScalars->GetNumberOfTuples(); i++)
	//{
	//	if (varScalars->GetValue(i) > maxScalar)
	//		maxScalar = varScalars->GetValue(i);
	//}

	//for (int i = 0; i < varScalars->GetNumberOfTuples(); i++)
	//{
	//	vtkFloatingPointType var = varScalars->GetValue(i);
	//	varScalars->SetValue(i, var / maxScalar);
	//}
	//// save to file
	//meanShape->GetPointData()->SetVectors(varVectors);
	//varShape->GetPointData()->SetVectors(varShapeVectors);
	//meanShape->GetPointData()->SetScalars(varScalars);
								  // Write the file
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer->SetFileName("test.vtp");
	writer->SetInputData(polyData);
	writer->Write();
	std::cout << "file write complete." << std::endl;
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

//C:\Users\Alex\Documents\shape_visual\build\bin\Release\itkIncrementalPCAModelEstimatorVisual.exe C:\Users\Alex\Documents\aligned\aligned.mvb 10 20 11