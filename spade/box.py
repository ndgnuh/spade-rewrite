from functools import cached_property
from enum import Enum
from dataclasses import dataclass


class BoxType(Enum):
    """
    Enum indicates type of bounding box representation.

    Warning
    ----------
    Convert from quadrilateral forms (XY4, X4Y4, QUAD) to
    rectangle forms (XXYY, XYXY, XYWH) might cause information loss.

    Class Attributes
    ----------
    XXYY:
        rectangle form: [x1, x2, y1, y2]
    XYXY:
        rectangle form: [x1, y1, x2, y2]
    XY4:
        quadrilateral form: [x1, y1, x2, y2, x3, y3, x4, y4]
    X4Y4:
        quadrilateral form: [x1, x2, x3, x4, y1, y2, y3, y4]
    QUAD:
        quadrilateral form: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    XYWH:
        rectangle form: [left, top, width, height]
    """
    XXYY = 0
    XYXY = 1
    XY4 = 2
    X4Y4 = 3
    QUAD = 4
    XYWH = 5


@dataclass(frozen=True, eq=True)
class Box:
    """A unified bounding box data structure

    This class can form bounding box from multiple format and
    can convert the current one to other types.

    See also: BoxType

    Attribute(s)
    ----------
    data: list
        the bounding box of any supported representation
    box_type: BoxType
        the enum indicate the type of bounding box
    xyxy: list
        represent the data in xyxy form, see `BoxType`
    xxyy: list
        same as xyxy
    xy4: list
        same as xyxy
    x4y4: list
        same as xyxy
    quad: list
        same as xyxy
    xywh: list
        same as xyxy

    Method(s)
    ----------
    map(box_type: BoxType) -> Box
        returns a new instance of Box with the internal representation
        of type `box_type`
    """
    data: tuple
    box_type: BoxType

    # Not a property
    # For easy access
    Type = BoxType

    # GET AROUND THE HASH WARNING
    def __hash__(self):
        return hash((str(self.data), self.box_type))

    # Create a new Box instance of another representation
    def map(self, box_type: BoxType):
        name = box_type.name.lower()
        data = getattr(self, name)
        return Box(data, box_type)

    # FEATURE PROPERTIES
    @cached_property
    def center(self):
        quad = self.quad
        xs = [c[0] for c in quad]
        ys = [c[1] for c in quad]
        n = len(quad) // 2
        return sum(xs) / n, sum(ys) / n

    # BASE CONVERSION
    # ALL OTHER CONVERSION ARE DERIVITIVE OF THESE TWO
    @cached_property
    def xyxy(self):
        bt = self.box_type
        data = self.data
        if bt == BoxType.XXYY:
            x1, x2, y1, y2 = data
            return [x1, y1, x2, y2]
        elif bt == BoxType.XYXY:
            return data
        elif bt == BoxType.XYWH:
            x1, y1, w, h = data
            x2 = w - x1
            y2 = h - y1
            return [x1, y1, x2, y2]
        elif bt == BoxType.XY4:
            n = len(data)
            x = [data[i] for i in range(0, n, 2)]
            y = [data[i] for i in range(1, n, 2)]
            return [min(x), min(y), max(x), max(y)]
        elif bt == BoxType.X4Y4:
            x = data[:4]
            y = data[4:]
            return [min(x), min(y), max(x), max(y)]
        elif bt == BoxType.QUAD:
            x = [d[0] for d in data]
            y = [d[1] for d in data]
            return [min(x), min(y), max(x), max(y)]

    @cached_property
    def quad(self):
        bt = self.box_type
        data = self.data
        if bt == BoxType.XXYY:
            x1, x2, y1, y2 = data
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif bt == BoxType.XYXY:
            x1, y1, x2, y2 = data
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif bt == BoxType.XYWH:
            x1, y1, w, h = data
            x2 = w - x1
            y2 = h - y1
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif bt == BoxType.XY4:
            n = len(data)
            xs = [data[i] for i in range(0, n, 2)]
            ys = [data[i] for i in range(1, n, 2)]
            return [[x, y] for (x, y) in zip(xs, ys)]
        elif bt == BoxType.X4Y4:
            xs = data[:4]
            ys = data[4:]
            return [[x, y] for (x, y) in zip(xs, ys)]
        elif bt == BoxType.QUAD:
            return data

    # DERIVED CONVERSION
    @cached_property
    def xxyy(self):
        x1, y1, x2, y2 = self.xyxy
        return x1, x2, y1, y2

    @cached_property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1

    @cached_property
    def xy4(self):
        quad = self.quad
        x0, y0 = quad[0]
        x1, y1 = quad[1]
        x2, y2 = quad[2]
        x3, y3 = quad[3]
        return [x0, y0, x1, y1, x2, y2, x3, y3]

    @cached_property
    def x4y4(self):
        quad = self.quad
        x0, y0 = quad[0]
        x1, y1 = quad[1]
        x2, y2 = quad[2]
        x3, y3 = quad[3]
        return [x0, x1, x2, x3, y0, y1, y2, y3]
