"""Open inspection cockpit: seat, dashboard, pedals and driving display."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def make_display_images(directory: Path):
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=True)
    w, h = 1600, 800
    image = Image.new("RGB", (w, h))
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = 330
    for y in range(h):
        if y < horizon:
            f = y / horizon
            color = np.array([78, 146, 197]) * (1-f) + np.array([192, 214, 221]) * f
        else:
            f = (y-horizon)/(h-horizon)
            color = np.array([88, 116, 77]) * (1-f) + np.array([48, 76, 47]) * f
        pixels[y] = color
    image = Image.fromarray(pixels)
    d = ImageDraw.Draw(image)
    d.polygon([(0,330),(170,220),(350,320),(590,235),(820,330)],fill=(105,132,137))
    d.polygon([(650,330),(900,250),(1130,315),(1370,240),(1600,330)],fill=(116,142,145))
    d.polygon([(720,330),(880,330),(1530,800),(70,800)],fill=(69,73,76))
    d.polygon([(718,330),(728,330),(123,800),(98,800)],fill=(232,231,216))
    d.polygon([(872,330),(882,330),(1500,800),(1475,800)],fill=(232,231,216))
    for start, end in [(347,360),(372,395),(418,453),(486,545),(593,687),(751,800)]:
        def point(y, lane, width):
            f=(y-330)/470
            return (800+lane*f*370+width*f,y)
        for lane in (-.65,.65):
            d.polygon([point(start,lane,-5),point(start,lane,5),point(end,lane,5),point(end,lane,-5)],fill=(236,235,218))
    for side in (-1,1):
        for j in range(1,12):
            f=(j/12)**1.6
            y=int(335+f*400);x=int(800+side*(95+f*690));r=int(8+f*44)
            d.rectangle((x-3,y-r,x+3,y+r),fill=(91,76,59))
            d.ellipse((x-r,y-r*3,x+r,y-r),fill=(42+int(f*12),87+int(f*10),58))
        for j in range(1,15):
            f=(j/15)**1.8;y=int(337+f*440);x=int(800+side*(86+f*680))
            d.rectangle((x-2-int(f*4),y-int(4+f*26),x+2+int(f*4),y),fill=(209,213,205))
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(font_path,24)
    d.rounded_rectangle((25,25,330,74),12,fill=(24,39,48))
    d.text((43,36),"DRIVING SIMULATOR",font=font,fill=(228,239,242))
    image.save(directory / "driving_view.png")
    cluster=Image.new("RGB",(1024,360),(10,17,22));d=ImageDraw.Draw(cluster)
    large=ImageFont.truetype(font_path,100);small=ImageFont.truetype(font_path,28)
    for x in (190,834):
        d.arc((x-130,40,x+130,300),145,395,fill=(50,180,208),width=12)
        for angle in np.linspace(145,395,11):
            a=np.deg2rad(angle);xx=x+115*np.cos(a);yy=170+115*np.sin(a)
            d.ellipse((xx-3,yy-3,xx+3,yy+3),fill=(205,219,225))
    d.text((478,85),"0",font=large,fill=(231,239,241))
    d.text((463,205),"km/h",font=small,fill=(127,161,173))
    d.text((170,145),"P",font=large,fill=(229,232,234))
    d.text((752,145),"READY",font=small,fill=(80,204,168))
    cluster.save(directory / "instrument_cluster.png")


def add_cabin(world, asset, directory, vec, quat):
    make_display_images(directory)
    for name, filename in [("driving", "driving_view.png"), ("instruments", "instrument_cluster.png")]:
        ET.SubElement(asset,"texture",name=name+"_texture",type="2d",file="../scenes/textures/"+filename)
        ET.SubElement(asset,"material",name=name+"_material",texture=name+"_texture",texuniform="false",emission=".6",specular="0")
    cabin=ET.SubElement(world,"body",name="cockpit")
    def box(name,pos,size,color,**kw):
        return ET.SubElement(cabin,"geom",name=name,type="box",pos=vec(pos),size=vec(size),
                             rgba=color,contype="4",conaffinity="2",**kw)
    def bar(name,a,b,r,color):
        return ET.SubElement(cabin,"geom",name=name,type="capsule",fromto=vec([*a,*b]),size=str(r),
                             rgba=color,contype="4",conaffinity="2")
    dark=".055 .065 .08 1";trim=".19 .22 .26 1";seat=".075 .085 .105 1"
    box("cockpit_floor",[.45,0,.11],[.98,.70,.045],dark)
    for y in (-.66,.66):
        box(f"side_sill_{y}",[.40,y,.22],[.95,.045,.075],trim)
        box(f"door_inner_{y}",[.42,y,.59],[.82,.035,.30],dark)
        box(f"door_insert_{y}",[.25,y*.97,.66],[.42,.018,.12],".105 .12 .145 1")
        box(f"door_armrest_{y}",[.20,y*.90,.78],[.30,.065,.035],seat)
        bar(f"window_sill_{y}",[-.34,y,.91],[1.13,y,.91],.025,trim)
        bar(f"a_pillar_{y}",[1.05,y,.89],[1.45,y*.96,1.86],.035,trim)
        bar(f"rear_pillar_{y}",[-.48,y,.25],[-.48,y,1.83],.028,trim)
        bar(f"roof_rail_{y}",[-.48,y,1.83],[1.45,y*.96,1.86],.025,trim)
    bar("windshield_header",[1.45,-.64,1.86],[1.45,.64,1.86],.025,trim)
    box("seat_pedestal",[-.03,0,.45],[.19,.23,.23],".16 .18 .21 1")
    for y in (-.20,.20):
        box(f"seat_rail_{y}",[0,y,.68],[.28,.025,.025],trim)
    box("seat_cushion",[.015,0,.72],[.265,.255,.055],seat)
    for y in (-.245,.245):
        bar(f"seat_lower_bolster_{y}",[-.19,y,.765],[.22,y,.765],.035,seat)
    box("seat_back",[-.245,0,1.04],[.055,.255,.285],seat,euler="0 -.08 0")
    box("seat_back_insert",[-.183,0,1.075],[.012,.19,.19],".115 .13 .155 1",euler="0 -.08 0")
    box("seat_cushion_insert",[.015,0,.778],[.20,.185,.008],".115 .13 .155 1")
    for z in (.94,1.01,1.08,1.15,1.22):
        bar(f"seat_stitch_{z}",[-.167,-.17,z],[-.167,.17,z],.0014,".25 .28 .32 1")
    for y in (-.24,.24):
        bar(f"seat_back_bolster_{y}",[-.20,y,.85],[-.235,y,1.28],.038,seat)
    ET.SubElement(cabin,"geom",name="head_rest",type="ellipsoid",pos="-.285 0 1.44",size=".065 .17 .11",
                  rgba=seat,contype="4",conaffinity="2")
    for y in (-.095,.095):
        bar(f"headrest_post_{y}",[-.265,y,1.27],[-.28,y,1.39],.009,".55 .59 .62 1")
    # Lower dashboard keeps the horizontal G29 mounting deck visible.
    box("dashboard_lower",[1.03,0,.78],[.16,.58,.18],dark)
    box("dashboard_upper",[1.04,0,1.015],[.17,.59,.045],".07 .08 .095 1")
    bar("dashboard_trim",[.86,-.57,.985],[.86,.57,.985],.009,".45 .49 .54 1")
    box("center_console",[.58,-.445,.53],[.36,.105,.17],dark)
    box("console_lid",[.40,-.445,.72],[.18,.10,.025],seat)
    bar("gear_selector_stalk",[.81,-.445,.70],[.79,-.445,.80],.012,trim)
    ET.SubElement(cabin,"geom",name="gear_selector",type="ellipsoid",pos=".79 -.445 .805",size=".035 .025 .028",
                  rgba=seat,contype="4",conaffinity="2")
    for y in (-.485,-.405):
        ET.SubElement(cabin,"geom",name=f"cup_holder_{y}",type="cylinder",pos=vec([.68,y,.706]),
                      size=".032 .005",rgba=".018 .023 .03 1",contype="0",conaffinity="0")
    box("foot_plate",[.47,0,.22],[.23,.35,.025],".13 .15 .18 1")
    for name,y,size in [("accelerator",-.16,[.026,.038,.075]),("brake",-.035,[.027,.065,.065])]:
        box(name,[.67,y,.30],size,".28 .30 .33 1",euler="0 .32 0")
        for z in (.275,.30,.325):
            box(f"{name}_rib_{z}",[.642,y,z],[.006,size[1]*.85,.004],".10 .12 .14 1")
    # Wide monitor and instrument panel face the driver (-world X).
    face=Rotation.from_matrix(np.array([[0,0,-1],[-1,0,0],[0,1,0]],dtype=float))
    box("display_bezel",[1.58,0,1.43],[.025,.75,.385],".022 .028 .036 1")
    ET.SubElement(cabin,"geom",name="front_display",type="plane",pos="1.552 0 1.43",size=".72 .355 .001",
                  quat=quat(face),material="driving_material",contype="0",conaffinity="0")
    box("display_stand",[1.61,0,.64],[.04,.07,.42],trim)
    box("display_foot",[1.61,0,.19],[.20,.32,.03],trim)
    bar("rearview_mirror_stem",[1.43,0,1.83],[1.32,0,1.73],.008,dark)
    box("rearview_mirror",[1.31,0,1.72],[.016,.105,.035],dark)
    box("rearview_mirror_glass",[1.292,0,1.72],[.002,.093,.026],".26 .34 .40 1")
    box("instrument_bezel",[.865,0,1.075],[.025,.205,.085],dark)
    ET.SubElement(cabin,"geom",name="instrument_display",type="plane",pos=".837 0 1.075",size=".19 .067 .001",
                  quat=quat(face),material="instruments_material",contype="0",conaffinity="0")
    for side in (-1,1):
        for offset in (-.025,0,.025):
            box(f"vent_{side}_{offset}",[.858,side*.43+offset,1.02],[.006,.003,.024],".015 .02 .025 1")
