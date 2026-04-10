import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';
import axios from 'axios';
import { UploadCloud, FileType, CheckCircle, Loader2 } from 'lucide-react';
import './App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [projectName, setProjectName] = useState('');
  const [clientName, setClientName] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles) => {
    setFiles(acceptedFiles);
    setResults(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    noClick: true,
  });

  // Helper: Format date string to MM-DD-YYYY
  const formatTableDate = (dateStr) => {
    if (!dateStr) return '';
    try {
      const d = new Date(dateStr);
      if (isNaN(d.getTime())) return dateStr;
      const mm = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const yyyy = d.getFullYear();
      return `${mm}-${dd}-${yyyy}`;
    } catch {
      return dateStr;
    }
  };

  // Helper: Parse date for sorting
  const getSortableDate = (dateStr) => {
    if (!dateStr) return 0;
    const d = new Date(dateStr);
    return isNaN(d.getTime()) ? 0 : d.getTime();
  };

  const handleUploadAndProcess = async () => {
    if (files.length === 0) return;
    
    setLoading(true);
    setProgress(0);
    setResults(null);
    
    const allResults = {};

    try {
      for (let i = 0; i < files.length; i++) {
        const formData = new FormData();
        formData.append('files', files[i]);

        const response = await axios.post('http://localhost:8001/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        Object.assign(allResults, response.data);
        
        const currentProgress = Math.round(((i + 1) / files.length) * 100);
        setProgress(currentProgress);
      }

      setResults(allResults);
      await generateExcel(allResults);
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Failed to process PDFs. Please make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const generateExcel = async (data) => {
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Transmittal Report');

    // 1. HEADER SECTION (Styling and Merges)
    worksheet.mergeCells('A1:B3');
    worksheet.getCell('A1').value = 'CD';
    worksheet.getCell('A1').font = { name: 'Arial Black', size: 48, bold: true, color: { argb: 'FF003399' } };
    worksheet.getCell('A1').alignment = { vertical: 'middle', horizontal: 'center' };

    worksheet.mergeCells('C1:F3');
    worksheet.getCell('C1').value = 'CALDIM ENGINEERING PRIVATE LIMITED';
    worksheet.getCell('C1').font = { name: 'Arial', size: 24, bold: true, color: { argb: 'FF000000' } };
    worksheet.getCell('C1').alignment = { vertical: 'middle', horizontal: 'left' };

    const t_no = `#${Math.floor(Math.random() * 900) + 100}`;
    const today = formatTableDate(new Date());

    let extractedProjectNo = 'N/A';
    Object.values(data).some(docResults => {
      if (docResults && docResults.project_no && docResults.project_no !== 'N/A') {
        extractedProjectNo = docResults.project_no;
        return true;
      }
      return false;
    });

    const projectLabelStyle = { name: 'Arial', size: 12, bold: true, color: { argb: 'FF006633' } };
    
    worksheet.mergeCells('A4:D4');
    worksheet.getCell('A4').value = `PROJECT NAME : ${projectName.toUpperCase() || 'N/A'}`;
    worksheet.getCell('A4').font = projectLabelStyle;
    
    worksheet.mergeCells('E4:F4');
    worksheet.getCell('E4').value = `TRANSMITTAL NO: ${t_no}`;
    worksheet.getCell('E4').font = projectLabelStyle;
    worksheet.getCell('E4').alignment = { horizontal: 'right' };

    worksheet.mergeCells('A5:D5');
    worksheet.getCell('A5').value = `PROJECT NO        : ${extractedProjectNo}`;
    worksheet.getCell('A5').font = projectLabelStyle;

    worksheet.mergeCells('E5:F5');
    worksheet.getCell('E5').value = `Date: ${today}`;
    worksheet.getCell('E5').font = projectLabelStyle;
    worksheet.getCell('E5').alignment = { horizontal: 'right' };

    worksheet.mergeCells('A6:D6');
    worksheet.getCell('A6').value = `FABRICATOR       : ${clientName.toUpperCase() || 'N/A'}`;
    worksheet.getCell('A6').font = projectLabelStyle;

    worksheet.addRow([]);

    const headerRow = worksheet.addRow(['Sl. No.', 'DrawingNo.', 'Drawing Description', 'REV#', 'DATE', 'Remarks']);
    headerRow.font = { bold: true };
    headerRow.eachCell((cell) => {
      cell.border = { top: {style:'thin'}, left: {style:'thin'}, bottom: {style:'thin'}, right: {style:'thin'} };
      cell.alignment = { horizontal: 'center' };
    });

    const folderGroups = {};
    Object.keys(data).forEach((filename) => {
      const fileObj = files.find(f => f.name === filename);
      // Handle both / and \ for Windows compatibility
      const pathParts = (fileObj?.path || filename).split(/[/\\]/);
      // If pathParts has at least 2 elements (folder + file), get the last folder
      const folderHeader = pathParts.length >= 2 ? pathParts[pathParts.length - 2] : 'DOCUMENTS';
      
      if (!folderGroups[folderHeader]) folderGroups[folderHeader] = [];
      folderGroups[folderHeader].push({ filename, results: data[filename] });
    });

    let slNo = 1;
    Object.keys(folderGroups).forEach(folderName => {
      const folderRow = worksheet.addRow(['', '', folderName.toUpperCase(), '', '', '']);
      worksheet.mergeCells(folderRow.number, 3, folderRow.number, 6);
      folderRow.getCell(3).font = { name: 'Arial', size: 11, bold: true, color: { argb: 'FFFF0000' } };
      folderRow.getCell(3).alignment = { horizontal: 'center' };

      folderGroups[folderName].forEach((fileData) => {
        let drawingNo = fileData.results.drawing_no || '';
        let drawingDesc = fileData.results.drawing_description || '';
        let revRows = fileData.results.revisions || [];

        const latest = fileData.results.latest_revision || { rev: '0', date: today, remarks: 'ISSUED FOR FABRICATION' };

        const row = worksheet.addRow([
          slNo++,
          drawingNo,
          drawingDesc,
          latest.rev,
          formatTableDate(latest.date),
          latest.remarks
        ]);

        const bgColor = (slNo % 2 === 0) ? 'FFF2F2F2' : 'FFFFFFFF';
        row.eachCell((cell) => {
          cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: bgColor } };
          cell.border = { top: {style:'thin'}, left: {style:'thin'}, bottom: {style:'thin'}, right: {style:'thin'} };
        });
      });
    });

    worksheet.getColumn(1).width = 8;
    worksheet.getColumn(2).width = 18;
    worksheet.getColumn(3).width = 60;
    worksheet.getColumn(4).width = 10;
    worksheet.getColumn(5).width = 15;
    worksheet.getColumn(6).width = 35;

    const buffer = await workbook.xlsx.writeBuffer();
    saveAs(new Blob([buffer]), `Transmittal_${projectName || 'Report'}.xlsx`);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>AI Document Text Extraction</h1>
        <p>Premium Project & Drawing Management</p>
      </header>
      
      <main className="main-content">
        <div className="project-card">
          <h3>Project Details</h3>
          <div className="project-form">
            <div className="input-group">
              <label>Project Name</label>
              <input 
                type="text" 
                placeholder="e.g. BURJ KHALIFA SITE A" 
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
              />
            </div>
            <div className="input-group">
              <label>Client Name (Fabricator)</label>
              <input 
                type="text" 
                placeholder="e.g. STEEL FAB ENTERPRISES" 
                value={clientName}
                onChange={(e) => setClientName(e.target.value)}
              />
            </div>
          </div>
        </div>

        <div 
          {...getRootProps()} 
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps({ webkitdirectory: "true" })} />
          <UploadCloud className="upload-icon" size={48} />
          {isDragActive ? (
            <p className="drop-text">Drop the folders here...</p>
          ) : (
            <>
              <p className="drop-text">
                Drag & drop folders here, or 
                <span className="browse-link" onClick={open}> select folders</span>
              </p>
            </>
          )}
          <p className="sub-text">Subfolders (e.g. E-SHEETS, D-SHEETS) will be used as groups in the Excel.</p>
        </div>

        {files.length > 0 && (
          <div className="file-list-container">
            <h3>Selected Drawings ({files.length})</h3>
            <ul className="file-list">
              {files.map((file, idx) => (
                <li key={idx} className="file-item">
                  <FileType className="file-icon" size={20} />
                  <div className="file-info">
                    <span className="file-name">{file.name}</span>
                    <span className="folder-path">{file.path || file.name}</span>
                  </div>
                </li>
              ))}
            </ul>
            
            <button 
              className={`process-button ${loading ? 'loading' : ''}`} 
              onClick={handleUploadAndProcess}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="spinner" size={20} />
                  Processing...
                </>
              ) : (
                'Generate Premium Excel Report'
              )}
            </button>

            {loading && (
              <div className="progress-container">
                <div className="progress-text">
                  <span>Processing Files</span>
                  <span>{progress}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}

        {results && (
          <div className="success-banner">
            <CheckCircle className="success-icon" size={24} />
            <div>
              <h3>Extraction Complete!</h3>
              <p>Your premium Excel report has been downloaded with professional formatting.</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
