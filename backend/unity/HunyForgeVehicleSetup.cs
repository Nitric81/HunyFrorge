// HunyForge Vehicle Setup — creates WheelColliders + Rigidbody for a rigged
// vehicle GLB produced by HunyForge (Chassis + Wheel_* node hierarchy).
// Usage: import the unity package, drop the vehicle prefab into a scene,
// select it, then run HunyForge → Setup Vehicle Colliders.
#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

public static class HunyForgeVehicleSetup
{
    [Serializable]
    private class WheelSpec
    {
        public string name;
        public float[] center;
        public float radius;
        public float half_width;
        public bool steer;
    }

    [Serializable]
    private class Suggested
    {
        public float suspension_distance = 0.1f;
        public float wheel_mass = 20f;
        public float chassis_mass = 1200f;
    }

    [Serializable]
    private class VehicleBlock
    {
        public WheelSpec[] wheels;
        public string chassis_node = "Chassis";
        public Suggested suggested = new Suggested();
    }

    [Serializable]
    private class Manifest
    {
        public VehicleBlock vehicle;
    }

    [MenuItem("HunyForge/Setup Vehicle Colliders")]
    public static void SetupSelected()
    {
        var root = Selection.activeGameObject;
        if (root == null)
        {
            Debug.LogError("[HunyForge] Select the imported vehicle prefab instance in the scene first.");
            return;
        }
        var manifestPath = FindManifest();
        if (manifestPath == null)
        {
            Debug.LogError("[HunyForge] hunyforge-manifest.json not found under Assets/.");
            return;
        }
        var manifest = JsonUtility.FromJson<Manifest>(File.ReadAllText(manifestPath));
        if (manifest?.vehicle?.wheels == null || manifest.vehicle.wheels.Length == 0)
        {
            Debug.LogError("[HunyForge] Manifest has no vehicle block — this package was not produced by the vehicle rigging flow.");
            return;
        }
        var suggested = manifest.vehicle.suggested ?? new Suggested();
        var rb = root.GetComponent<Rigidbody>();
        if (rb == null) rb = root.AddComponent<Rigidbody>();
        rb.mass = Mathf.Max(1f, suggested.chassis_mass);
        var container = root.transform.Find("WheelColliders");
        if (container == null)
        {
            container = new GameObject("WheelColliders").transform;
            container.SetParent(root.transform, false);
        }
        var created = 0;
        foreach (var spec in manifest.vehicle.wheels)
        {
            var visual = FindDeepChild(root.transform, spec.name);
            if (visual == null)
            {
                Debug.LogWarning($"[HunyForge] Wheel node '{spec.name}' not found under {root.name} — skipped.");
                continue;
            }
            var holderName = spec.name + "_Collider";
            var existing = container.Find(holderName);
            var holder = existing != null ? existing.gameObject : new GameObject(holderName);
            holder.transform.SetParent(container, false);
            holder.transform.position = visual.position;
            var scale = Mathf.Max(0.0001f, visual.lossyScale.x);
            var wc = holder.GetComponent<WheelCollider>();
            if (wc == null) wc = holder.AddComponent<WheelCollider>();
            wc.radius = Mathf.Max(0.01f, spec.radius * scale);
            wc.mass = Mathf.Max(0.1f, suggested.wheel_mass);
            wc.suspensionDistance = Mathf.Max(0.001f, suggested.suspension_distance);
            var spring = wc.suspensionSpring;
            spring.spring = Mathf.Max(1000f, suggested.chassis_mass * 25f);
            spring.damper = Mathf.Max(100f, suggested.chassis_mass * 2f);
            spring.targetPosition = 0.5f;
            wc.suspensionSpring = spring;
            created++;
        }
        Debug.Log($"[HunyForge] Configured {created} WheelColliders on '{root.name}' (mass {rb.mass:F0} kg). Front wheels steer: mark per WheelCollider as needed.");
        Selection.activeGameObject = root;
    }

    private static string FindManifest()
    {
        var direct = Path.Combine(Application.dataPath, "HunyForge/hunyforge-manifest.json");
        if (File.Exists(direct)) return direct;
        foreach (var path in Directory.GetFiles(Application.dataPath, "hunyforge-manifest.json", SearchOption.AllDirectories))
            return path;
        return null;
    }

    private static Transform FindDeepChild(Transform parent, string name)
    {
        if (parent.name == name) return parent;
        for (var i = 0; i < parent.childCount; i++)
        {
            var found = FindDeepChild(parent.GetChild(i), name);
            if (found != null) return found;
        }
        return null;
    }
}
#endif
