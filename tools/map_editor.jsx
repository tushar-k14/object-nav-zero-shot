import { useState, useRef, useCallback, useEffect } from "react";

const ROOM_COLORS = {
  bedroom: "#FF6B6B",
  bathroom: "#4ECDC4",
  kitchen: "#FFE66D",
  living_room: "#95E1D3",
  dining_room: "#F38181",
  hallway: "#AA96DA",
  office: "#6C5CE7",
  garage: "#A8D8EA",
  laundry: "#FCBAD3",
  closet: "#DDA0DD",
  other: "#888888",
};

const TOOLS = [
  { id: "room_polygon", label: "Draw Room", icon: "⬡" },
  { id: "mark_object", label: "Mark Object", icon: "◎" },
  { id: "set_center", label: "Set Room Center", icon: "✛" },
  { id: "erase", label: "Erase", icon: "✕" },
  { id: "pan", label: "Pan / Select", icon: "✋" },
];

const ROOM_TYPES = [
  "bedroom", "bathroom", "kitchen", "living_room",
  "dining_room", "hallway", "office", "garage",
  "laundry", "closet", "other",
];

const OBJECT_TYPES = [
  "bed", "chair", "toilet", "sofa", "tv_monitor",
  "plant", "table", "refrigerator", "sink", "lamp",
  "desk", "microwave", "oven", "custom",
];

export default function MapEditor() {
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [mapImage, setMapImage] = useState(null);
  const [mapSize, setMapSize] = useState({ w: 800, h: 600 });
  const [tool, setTool] = useState("room_polygon");
  const [roomType, setRoomType] = useState("bedroom");
  const [objectType, setObjectType] = useState("bed");
  const [customLabel, setCustomLabel] = useState("");

  // Annotation state
  const [rooms, setRooms] = useState([]);
  const [objects, setObjects] = useState([]);
  const [currentPolygon, setCurrentPolygon] = useState([]);
  const [selectedRoom, setSelectedRoom] = useState(null);

  // Viewport
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const toCanvas = useCallback((clientX, clientY) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left - offset.x) / zoom,
      y: (clientY - rect.top - offset.y) / zoom,
    };
  }, [zoom, offset]);

  // Load map image
  const handleLoadImage = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      setMapImage(img);
      setMapSize({ w: img.width, h: img.height });
      setZoom(Math.min(800 / img.width, 600 / img.height, 1));
      setOffset({ x: 0, y: 0 });
    };
    img.src = URL.createObjectURL(file);
  };

  // Canvas click handler
  const handleCanvasClick = (e) => {
    const pos = toCanvas(e.clientX, e.clientY);
    const ix = Math.round(pos.x);
    const iy = Math.round(pos.y);

    if (tool === "room_polygon") {
      setCurrentPolygon(prev => [...prev, { x: ix, y: iy }]);
    }
    else if (tool === "mark_object") {
      const label = objectType === "custom" ? (customLabel || "unknown") : objectType;
      setObjects(prev => [...prev, {
        id: `obj_${Date.now()}`,
        label,
        position: { x: ix, y: iy },
        room: selectedRoom,
      }]);
    }
    else if (tool === "set_center") {
      if (selectedRoom !== null) {
        setRooms(prev => prev.map((r, i) =>
          i === selectedRoom ? { ...r, center: { x: ix, y: iy } } : r
        ));
      }
    }
    else if (tool === "erase") {
      // Check objects first
      const objIdx = objects.findIndex(o =>
        Math.abs(o.position.x - ix) < 15 && Math.abs(o.position.y - iy) < 15
      );
      if (objIdx >= 0) {
        setObjects(prev => prev.filter((_, i) => i !== objIdx));
        return;
      }
      // Check rooms
      const roomIdx = rooms.findIndex(r =>
        r.center && Math.abs(r.center.x - ix) < 20 && Math.abs(r.center.y - iy) < 20
      );
      if (roomIdx >= 0) {
        setRooms(prev => prev.filter((_, i) => i !== roomIdx));
        if (selectedRoom === roomIdx) setSelectedRoom(null);
      }
    }
    else if (tool === "pan") {
      // Select room by clicking near center
      const roomIdx = rooms.findIndex(r =>
        r.center && Math.abs(r.center.x - ix) < 25 && Math.abs(r.center.y - iy) < 25
      );
      if (roomIdx >= 0) setSelectedRoom(roomIdx);
      else setSelectedRoom(null);
    }
  };

  // Finish polygon (double-click or Enter)
  const finishPolygon = useCallback(() => {
    if (currentPolygon.length < 3) return;
    const cx = Math.round(currentPolygon.reduce((s, p) => s + p.x, 0) / currentPolygon.length);
    const cy = Math.round(currentPolygon.reduce((s, p) => s + p.y, 0) / currentPolygon.length);
    const newRoom = {
      id: `room_${Date.now()}`,
      type: roomType,
      label: `${roomType}_${rooms.filter(r => r.type === roomType).length + 1}`,
      polygon: [...currentPolygon],
      center: { x: cx, y: cy },
    };
    setRooms(prev => [...prev, newRoom]);
    setSelectedRoom(rooms.length);
    setCurrentPolygon([]);
  }, [currentPolygon, roomType, rooms]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Enter") finishPolygon();
      if (e.key === "Escape") setCurrentPolygon([]);
      if (e.key === "z" && e.ctrlKey) {
        setCurrentPolygon(prev => prev.slice(0, -1));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [finishPolygon]);

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cw = 800, ch = 600;
    canvas.width = cw;
    canvas.height = ch;

    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, cw, ch);

    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(zoom, zoom);

    // Draw map image
    if (mapImage) {
      ctx.drawImage(mapImage, 0, 0);
    } else {
      ctx.fillStyle = "#2d2d44";
      ctx.fillRect(0, 0, mapSize.w, mapSize.h);
      ctx.fillStyle = "#555";
      ctx.font = "20px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Load a map image to begin", mapSize.w / 2, mapSize.h / 2);
    }

    // Draw room polygons
    rooms.forEach((room, idx) => {
      const color = ROOM_COLORS[room.type] || ROOM_COLORS.other;
      ctx.fillStyle = color + "40";
      ctx.strokeStyle = idx === selectedRoom ? "#fff" : color;
      ctx.lineWidth = idx === selectedRoom ? 3 / zoom : 1.5 / zoom;

      if (room.polygon.length > 2) {
        ctx.beginPath();
        ctx.moveTo(room.polygon[0].x, room.polygon[0].y);
        room.polygon.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Room center + label
      if (room.center) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(room.center.x, room.center.y, 6 / zoom, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5 / zoom;
        ctx.stroke();

        ctx.fillStyle = "#fff";
        ctx.font = `bold ${12 / zoom}px monospace`;
        ctx.textAlign = "center";
        ctx.fillText(room.label, room.center.x, room.center.y - 10 / zoom);
      }
    });

    // Draw current polygon in progress
    if (currentPolygon.length > 0) {
      const color = ROOM_COLORS[roomType] || "#fff";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 / zoom;
      ctx.setLineDash([5 / zoom, 5 / zoom]);
      ctx.beginPath();
      ctx.moveTo(currentPolygon[0].x, currentPolygon[0].y);
      currentPolygon.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
      ctx.setLineDash([]);

      currentPolygon.forEach(p => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4 / zoom, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw objects
    objects.forEach(obj => {
      ctx.fillStyle = "#FFD700";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.5 / zoom;
      ctx.beginPath();
      ctx.arc(obj.position.x, obj.position.y, 5 / zoom, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#FFD700";
      ctx.font = `${10 / zoom}px monospace`;
      ctx.textAlign = "center";
      ctx.fillText(obj.label, obj.position.x, obj.position.y - 8 / zoom);
    });

    ctx.restore();
  }, [mapImage, mapSize, rooms, objects, currentPolygon, selectedRoom, zoom, offset, roomType]);

  // Export annotations
  const exportAnnotations = () => {
    const data = {
      map_size: mapSize,
      rooms: {},
      objects: [],
      metadata: {
        created: new Date().toISOString(),
        tool: "ObjectNav Map Editor v1",
      },
    };

    rooms.forEach(room => {
      data.rooms[room.label] = {
        type: room.type,
        center: [room.center.x, room.center.y],
        polygon: room.polygon.map(p => [p.x, p.y]),
      };
    });

    objects.forEach(obj => {
      data.objects.push({
        label: obj.label,
        position: [obj.position.x, obj.position.y],
        room: obj.room !== null && rooms[obj.room]
          ? rooms[obj.room].label : null,
      });
    });

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "map_annotations.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import annotations
  const importAnnotations = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        const loadedRooms = [];
        const loadedObjects = [];

        if (data.rooms) {
          Object.entries(data.rooms).forEach(([label, r]) => {
            loadedRooms.push({
              id: `room_${Date.now()}_${label}`,
              type: r.type,
              label,
              polygon: r.polygon.map(p => ({ x: p[0], y: p[1] })),
              center: { x: r.center[0], y: r.center[1] },
            });
          });
        }

        if (data.objects) {
          data.objects.forEach(obj => {
            loadedObjects.push({
              id: `obj_${Date.now()}_${obj.label}`,
              label: obj.label,
              position: { x: obj.position[0], y: obj.position[1] },
              room: null,
            });
          });
        }

        setRooms(loadedRooms);
        setObjects(loadedObjects);
      } catch (err) {
        alert("Invalid annotation file");
      }
    };
    reader.readAsText(file);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const newZoom = Math.max(0.2, Math.min(5, zoom - e.deltaY * 0.001));
    setZoom(newZoom);
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100vh",
      background: "#0f0f1a", color: "#e0e0e0",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      {/* Header */}
      <div style={{
        padding: "8px 16px", background: "#1a1a2e",
        borderBottom: "1px solid #333", display: "flex",
        alignItems: "center", gap: 16, flexShrink: 0,
      }}>
        <span style={{ fontWeight: "bold", color: "#4ECDC4", fontSize: 14 }}>
          ◈ MAP EDITOR
        </span>
        <button onClick={() => fileInputRef.current?.click()}
          style={btnStyle}>
          📁 Load Map
        </button>
        <input ref={fileInputRef} type="file" accept="image/*"
          onChange={handleLoadImage} style={{ display: "none" }} />
        <button onClick={exportAnnotations} style={btnStyle}>
          💾 Export JSON
        </button>
        <label style={btnStyle}>
          📂 Import JSON
          <input type="file" accept=".json"
            onChange={importAnnotations} style={{ display: "none" }} />
        </label>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "#666" }}>
          Rooms: {rooms.length} | Objects: {objects.length} | Zoom: {zoom.toFixed(1)}x
        </span>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Toolbar */}
        <div style={{
          width: 200, background: "#16162b", padding: 12,
          borderRight: "1px solid #333", overflowY: "auto", flexShrink: 0,
        }}>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 8 }}>TOOLS</div>
          {TOOLS.map(t => (
            <button key={t.id} onClick={() => setTool(t.id)}
              style={{
                ...toolBtnStyle,
                background: tool === t.id ? "#4ECDC455" : "transparent",
                borderColor: tool === t.id ? "#4ECDC4" : "#333",
              }}>
              <span style={{ fontSize: 16 }}>{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}

          {tool === "room_polygon" && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>ROOM TYPE</div>
              {ROOM_TYPES.map(r => (
                <button key={r} onClick={() => setRoomType(r)}
                  style={{
                    ...toolBtnStyle, padding: "4px 8px", fontSize: 11,
                    background: roomType === r ? (ROOM_COLORS[r] || "#888") + "33" : "transparent",
                    borderColor: roomType === r ? ROOM_COLORS[r] || "#888" : "#333",
                  }}>
                  <span style={{
                    width: 10, height: 10, borderRadius: 2,
                    background: ROOM_COLORS[r] || "#888", display: "inline-block",
                  }} />
                  <span>{r.replace("_", " ")}</span>
                </button>
              ))}
              {currentPolygon.length > 0 && (
                <div style={{ marginTop: 8, fontSize: 11, color: "#4ECDC4" }}>
                  {currentPolygon.length} points — Enter to finish, Esc to cancel
                </div>
              )}
            </div>
          )}

          {tool === "mark_object" && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>OBJECT TYPE</div>
              {OBJECT_TYPES.map(o => (
                <button key={o} onClick={() => setObjectType(o)}
                  style={{
                    ...toolBtnStyle, padding: "4px 8px", fontSize: 11,
                    background: objectType === o ? "#FFD70033" : "transparent",
                    borderColor: objectType === o ? "#FFD700" : "#333",
                  }}>
                  {o}
                </button>
              ))}
              {objectType === "custom" && (
                <input
                  value={customLabel}
                  onChange={e => setCustomLabel(e.target.value)}
                  placeholder="Custom label..."
                  style={{
                    width: "100%", marginTop: 6, padding: "4px 8px",
                    background: "#1a1a2e", border: "1px solid #444",
                    color: "#e0e0e0", borderRadius: 4, fontSize: 12,
                  }}
                />
              )}
            </div>
          )}

          {/* Room list */}
          {rooms.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>ROOMS</div>
              {rooms.map((r, i) => (
                <div key={r.id}
                  onClick={() => { setSelectedRoom(i); setTool("pan"); }}
                  style={{
                    padding: "4px 8px", marginBottom: 2, borderRadius: 4, fontSize: 11,
                    cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
                    background: selectedRoom === i ? "#ffffff15" : "transparent",
                  }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: 2,
                    background: ROOM_COLORS[r.type] || "#888",
                  }} />
                  {r.label}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Canvas */}
        <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            onDoubleClick={finishPolygon}
            onWheel={handleWheel}
            style={{ cursor: tool === "pan" ? "grab" : "crosshair", display: "block" }}
          />
          {!mapImage && (
            <div style={{
              position: "absolute", top: "50%", left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center", color: "#555",
            }}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🗺️</div>
              <div style={{ fontSize: 14 }}>Load an occupancy map image to start annotating</div>
              <div style={{ fontSize: 11, marginTop: 4, color: "#444" }}>
                Use trajectory PNG or any top-down map
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const btnStyle = {
  padding: "4px 12px", background: "#2d2d44", border: "1px solid #444",
  color: "#e0e0e0", borderRadius: 4, cursor: "pointer", fontSize: 12,
  display: "inline-flex", alignItems: "center", gap: 4,
};

const toolBtnStyle = {
  width: "100%", padding: "6px 10px", marginBottom: 4,
  background: "transparent", border: "1px solid #333",
  color: "#e0e0e0", borderRadius: 4, cursor: "pointer", fontSize: 12,
  display: "flex", alignItems: "center", gap: 8, textAlign: "left",
};
