import cv2
import numpy as np
from datetime import datetime
from xml.dom import minidom
import xml.etree.ElementTree as etree
from Utils import Bbox, LayoutData, get_utc_time


class PageXMLExporter:
    def __init__(self, output_dir: str) -> None:
       self.output_dir = output_dir


    def get_bbox(self, bbox: Bbox) -> tuple[int, int, int, int]:
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h

        return (x, y, w, h)


    def get_text_points(self, contour):
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    def get_bbox_points(self, bbox: tuple[int]):
        x, y, w, h = bbox
        points = f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"
        return points

    
    def get_text_line_block(self, coordinate, baseline_points, index, unicode_text):
        text_line = etree.Element(
            "Textline", id="", custom=f"readingOrder {{index:{index};}}"
        )
        text_line = etree.Element("TextLine")
        text_line_coords = coordinate

        text_line.attrib["id"] = f"line_9874_{str(index)}"
        text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

        coords_points = etree.SubElement(text_line, "Coords")
        coords_points.attrib["points"] = text_line_coords
        baseline = etree.SubElement(text_line, "Baseline")
        baseline.attrib["points"] = baseline_points

        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_field = etree.SubElement(text_equiv, "Unicode")
        unicode_field.text = unicode_text

        return text_line
    


    def build_xml_document(self,
        image: np.array,
        image_name: str,
        images: tuple[int],
        lines,
        margins: tuple[int],
        captions: tuple[int],
        text_region_bbox: tuple[int],
        text_lines: list[str] | None,
    ):
        root = etree.Element("PcGts")
        root.attrib[
            "xmlns"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        root.attrib[
            "xsi:schemaLocation"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

        metadata = etree.SubElement(root, "Metadata")
        creator = etree.SubElement(metadata, "Creator")
        creator.text = "Transkribus"
        created = etree.SubElement(metadata, "Created")
        created.text = get_utc_time()

        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}"
        page.attrib["imageHeight"] = f"{image.shape[0]}"

        reading_order = etree.SubElement(page, "ReadingOrder")
        ordered_group = etree.SubElement(reading_order, "OrderedGroup")
        ordered_group.attrib["id"] = f"1234_{0}"
        ordered_group.attrib["caption"] = "Regions reading order"

        region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
        region_ref_indexed.attrib["index"] = "0"
        region_ref = "region_main"
        region_ref_indexed.attrib["regionRef"] = region_ref

        text_region = etree.SubElement(page, "TextRegion")
        text_region.attrib["id"] = region_ref
        text_region.attrib["custom"] = "readingOrder {index:0;}"

        text_region_coords = etree.SubElement(text_region, "Coords")
        text_region_coords.attrib["points"] = self.get_bbox_points(text_region_bbox)


        def get_line_baseline(bbox: tuple[int, int, int, int]) -> str:  
                x, y, w, h = bbox
                
                return f"{x},{y+h} {x+w},{y+h}"

        for i in range(0, len(lines)):
            text_coords = self.get_bbox_points(lines[i])
            base_line_coords = get_line_baseline(lines[i])
            
            if text_lines != None and len(text_lines) > 0:
                text_region.append(
                    self.get_text_line_block(coordinate=text_coords, baseline_points=base_line_coords, index=i, unicode_text=text_lines[i])
                )
            else:
                text_region.append(self.get_text_line_block(coordinate=text_coords, baseline_points=base_line_coords, index=i, unicode_text=""))


        if len(images) > 0:
            for idx, bbox in enumerate(images):
                image_region = etree.SubElement(page, "ImageRegion")
                image_region.attrib["id"] = "Image_1234"
                image_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}}"

                coords_points = etree.SubElement(image_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)

        if len(margins) > 0:
            for idx, bbox in enumerate(margins):
                margin_region = etree.SubElement(page, "TextRegion")
                margin_region.attrib["id"] = f"margin_1234_{idx}"
                margin_region.attrib["type"] = "margin"
                margin_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}} structure {{type:marginalia;}}"

                coords_points = etree.SubElement(margin_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)
        
        if len(captions) > 0:
            for idx, bbox in enumerate(captions):
                captions_region = etree.SubElement(page, "TextRegion")
                captions_region.attrib["id"] = f"caption_1234_{idx}"
                captions_region.attrib["type"] = "caption"
                captions_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}} structure {{type:caption;}}"

                coords_points = etree.SubElement(captions_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)
        
        xmlparse = minidom.parseString(etree.tostring(root))
        prettyxml = xmlparse.toprettyxml()

        return prettyxml


    def export(self, image: np.array, image_name: str, layout_data: LayoutData, text_lines: list[str]):
        image_boxes = [self.get_bbox(x) for x in layout_data.images]
        caption_boxes = [self.get_bbox(x) for x in layout_data.captions]
        margin_boxes = [self.get_bbox(x) for x in layout_data.margins]
        line_boxes = [self.get_bbox(x.bbox) for x in layout_data.lines]
        text_bbox = self.get_bbox(layout_data.text_bboxes[0])

        xml_doc = self.build_xml_document(
            image, 
            image_name, 
            images=image_boxes, 
            lines=line_boxes, 
            margins=margin_boxes, 
            captions=caption_boxes, 
            text_region_bbox=text_bbox, 
            text_lines=text_lines
            )
        
        xml_out = f"{self.output_dir}/{image_name}.xml"
        with open(xml_out, "w", encoding="utf-8") as f:
            f.write(xml_doc)