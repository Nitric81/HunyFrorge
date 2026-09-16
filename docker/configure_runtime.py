from pathlib import Path
import sys


REPLACEMENTS = {
    'hy3dpaint/utils/multiview_utils.py': [
        ('        self.cfg = cfg\n', '        self.cfg = cfg\n        self.inference_steps = config.texture_inference_steps\n        self.guidance_scale = config.texture_guidance_scale\n        self.seed = config.seed\n        self.telemetry = config.telemetry\n        self.dino_device = config.dino_device\n'),
        ('        self.seed_everything(0)', '        self.seed_everything(self.seed)'),
        ('manual_seed(0)', 'manual_seed(self.seed)'),
        ('num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__]', 'num_inference_steps=self.inference_steps'),
        ('            guidance_scale=3.0,', '            guidance_scale=self.guidance_scale,\n            callback_on_step_end=self.progress_callback,'),
        ('    def seed_everything(self, seed):', '    def progress_callback(self, pipeline, step, timestep, callback_kwargs):\n        self.telemetry.update("texture_diffusion", {"step": step + 1, "total": self.inference_steps})\n        return callback_kwargs\n\n    def seed_everything(self, seed):'),
        ('            dino_hidden_states = self.dino_v2(dino_input).to(self.pipeline.device)', '            with self.telemetry.measure("dino_features"):\n                self.dino_v2.to(self.dino_device)\n                dino_hidden_states = self.dino_v2(dino_input).to(self.pipeline.device)\n                if self.dino_device == "cuda":\n                    self.dino_v2.to("cpu")\n                    torch.cuda.empty_cache()'),
        ('            torch_dtype=torch.float16\n        )', '            torch_dtype=torch.float16,\n            trust_remote_code=True\n        )'),
    ],
    'hy3dpaint/textureGenPipeline.py': [
        ('            remesh_mesh(mesh_path, processed_mesh_path)', '            remesh_mesh(mesh_path, processed_mesh_path, target_count=self.config.face_count)'),
        ('        for i in range(len(enhance_images)):', '        for i in range(len(enhance_images["albedo"])):'),
    ],
    'hy3dpaint/utils/pipeline_utils.py': [
        ('        for idx in range(6):', '        for idx in range(min(6, max_selected_view_num, candidate_view_num)):'),
    ],
    'hy3dpaint/utils/simplify_mesh_utils.py': [
        ('def remesh_mesh(mesh_path, remesh_path):', 'def remesh_mesh(mesh_path, remesh_path, target_count=40000):'),
        ('    mesh = mesh_simplify_trimesh(mesh_path, remesh_path)', '    mesh = mesh_simplify_trimesh(mesh_path, remesh_path, target_count=target_count)'),
        ('courent.simplify_quadric_decimation(target_count)', 'courent.simplify_quadric_decimation(face_count=target_count)'),
    ],
}


def transform(relative, content):
    for old, new in REPLACEMENTS[relative]:
        if content.count(old) != 1:
            raise ValueError(f'Pinned upstream source mismatch: {relative}: {old!r}')
        content = content.replace(old, new, 1)
    return content


def main(root):
    transformed = {relative: transform(relative, (root / relative).read_text(encoding='utf-8')) for relative in REPLACEMENTS}
    for relative, content in transformed.items():
        compile(content, relative, 'exec')
    for relative, content in transformed.items():
        (root / relative).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
