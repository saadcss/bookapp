import os
import sys
import fitz  # PyMuPDF
from pathlib import Path
import re

def extract_figures_and_tables(pdf_path, output_dir="extracted_content"):
    """
    Extract figures and tables from a PDF by capturing full pages as images
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory to save extracted content
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    # Create a new PDF for output
    output_pdf = fitz.open()
    
    # Get base filename
    base_name = Path(pdf_path).stem
    
    print(f"Processing: {pdf_path}")
    print(f"Total pages: {len(pdf_document)}")
    
    figures_tables_found = {}
    extracted_count = 0
    
    # Process each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        
        # Look for actual figure/table captions, not just references
        # Figure captions: "• Fig. X.Y" or line starting with "Fig. X.Y"
        # Also check for "• Fig. X.Y" pattern which indicates a caption
        has_figure = bool(re.search(r'(?:^|\n)(?:•\s*[Ff]ig|^\s*[Ff]ig)(?:ure)?\.?\s+\d+\.?\d*', text, re.MULTILINE))
        
        # Table headers: All caps "TABLE X.Y" which is typical for table titles
        has_table = bool(re.search(r'(?:^|\n)\s*TABLE\s+\d+\.?\d*', text, re.MULTILINE))
        
        if has_figure or has_table:
            content_items = []
            
            # Extract the actual caption text for naming
            if has_figure:
                fig_matches = re.findall(r'(?:^|\n)(?:•\s*)?([Ff]ig(?:ure)?\.?\s+\d+\.?\d*)', text, re.MULTILINE)
                for fig in fig_matches:
                    fig_clean = re.sub(r'[•\s]', '', fig).strip()
                    if fig_clean:
                        content_items.append(fig_clean)
            
            if has_table:
                table_matches = re.findall(r'(?:^|\n)\s*(TABLE\s+\d+\.?\d*)', text, re.MULTILINE)
                for table in table_matches:
                    table_clean = table.strip()
                    if table_clean:
                        content_items.append(table_clean)
            
            if content_items:
                content_label = "_".join(content_items[:3])  # Limit to first 3 items
                if len(content_items) > 3:
                    content_label += "_plus"
                
                print(f"\nPage {page_num + 1}: Found {', '.join(content_items)}")
                
                # Extract page at original PDF resolution (72 DPI standard)
                zoom = 3.0  # Original resolution
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False reduces file size
                
                # Compress to JPEG for smaller file size (quality 85 is good balance)
                img_bytes = pix.tobytes("jpeg", jpg_quality=100)
                
                # Save page image as JPEG for smaller file size
                content_filename = f"{base_name}_page{page_num+1:03d}_{content_label}.jpg"
                content_path = output_path / content_filename
                
                with open(content_path, "wb") as content_file:
                    content_file.write(img_bytes)
                
                print(f"  Saved: {content_filename}")
                
                # Add to output PDF - create a new page from the image
                img_pdf = fitz.open()  # Create temporary PDF
                img_page = img_pdf.new_page(width=pix.width, height=pix.height)
                img_page.insert_image(img_page.rect, stream=img_bytes)
                output_pdf.insert_pdf(img_pdf)
                img_pdf.close()
                
                # Track what we found
                figures_tables_found[page_num + 1] = content_items
                extracted_count += 1
    
    # Save the output PDF only if we found content
    if extracted_count > 0:
        output_pdf_path = output_path / f"{base_name}_figures_and_tables.pdf"
        output_pdf.save(output_pdf_path)
        print(f"\n✓ Output PDF saved: {output_pdf_path}")
    else:
        print(f"\n⚠ No figures or tables found in this document")
    
    output_pdf.close()
    pdf_document.close()
    
    print(f"\n✓ Extraction complete!")
    print(f"✓ Total pages with figures/tables: {extracted_count}")
    
    # Print summary of what was found
    if figures_tables_found:
        print(f"\nDetailed summary:")
        for page_num, content_items in sorted(figures_tables_found.items()):
            print(f"  Page {page_num}: {', '.join(content_items)}")
    
    return output_path


def process_single_file(pdf_path, output_folder="extracted_output"):
    """Process a single PDF file"""
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Processing single file: {pdf_path}")
    print(f"{'='*80}\n")
    
    extract_figures_and_tables(pdf_path, output_folder)
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Check the '{output_folder}' folder for results.")
    print(f"{'='*80}\n")


def process_all_files_in_folder(folder_path, output_base_folder="extracted_all_files"):
    """Process all PDF files in a folder"""
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in: {folder_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Found {len(pdf_files)} PDF files in: {folder_path}")
    print(f"{'='*80}\n")
    
    # Create main output folder
    Path(output_base_folder).mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    no_content = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, pdf_file)
        
        print(f"\n{'='*80}")
        print(f"Processing file {i}/{len(pdf_files)}: {pdf_file}")
        print(f"{'='*80}\n")
        
        try:
            # Create subfolder for each PDF's output
            pdf_output_folder = os.path.join(output_base_folder, Path(pdf_file).stem)
            result = extract_figures_and_tables(pdf_path, pdf_output_folder)
            
            # Check if any files were created
            if list(Path(pdf_output_folder).glob("*.png")):
                successful += 1
                print(f"\n✓ Successfully processed: {pdf_file}")
            else:
                no_content += 1
                print(f"\n⊘ No figures/tables found in: {pdf_file}")
                
        except Exception as e:
            failed += 1
            print(f"\n✗ Error processing {pdf_file}: {str(e)}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total files: {len(pdf_files)}")
    print(f"Successfully extracted: {successful}")
    print(f"No content found: {no_content}")
    print(f"Failed: {failed}")
    print(f"Output location: {output_base_folder}/")
    print(f"{'='*80}\n")


def print_usage():
    """Print usage instructions"""
    print("="*80)
    print("PDF FIGURE AND TABLE EXTRACTOR")
    print("="*80)
    print("\nUsage:")
    print("  python pdf_extractor.py file <pdf_file_path> [output_folder]")
    print("  python pdf_extractor.py folder <folder_path> [output_folder]")
    print("\nExamples:")
    print('  python pdf_extractor.py file "chapter48.pdf"')
    print('  python pdf_extractor.py file "chapter48.pdf" "chapter48_output"')
    print('  python pdf_extractor.py folder "out_no_refs"')
    print('  python pdf_extractor.py folder "out_no_refs" "all_chapters_output"')
    print("\nOutput:")
    print("  - Individual PNG files for each page containing figures/tables")
    print("  - One compiled PDF with all extracted pages")
    print("  - Files named: ChapterName_pageXXX_Fig.X.Y_TableX.Y.png")
    print("\nOptions:")
    print("  file    - Process a single PDF file")
    print("  folder  - Process all PDF files in a folder")
    print("="*80)


def main():
    """Main function to handle command line arguments"""
    
    # Check if arguments are provided
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    path = sys.argv[2]
    
    # Get output folder if provided, otherwise use default
    if len(sys.argv) >= 4:
        output_folder = sys.argv[3]
    else:
        output_folder = None
    
    if mode == "file":
        # Process single file
        if output_folder is None:
            output_folder = "extracted_output"
        process_single_file(path, output_folder)
        
    elif mode == "folder":
        # Process all files in folder
        if output_folder is None:
            output_folder = "extracted_all_files"
        process_all_files_in_folder(path, output_folder)
        
    else:
        print(f"Error: Invalid mode '{mode}'. Use 'file' or 'folder'.")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
