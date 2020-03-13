#include <OpenMesh/Core/IO/MeshIO.hh>

#include <MishMesh/TriMesh.h>
#include <MishMesh/utils.h>
#include <MishMesh/dijkstra.h>
#include <MishMesh/geodesics.h>
#include <MishMesh/visualization.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>

#include <igl/exact_geodesic.h>

uint left_view, right_view;
uint left_mesh, right_mesh;

void set_points(igl::opengl::glfw::Viewer &viewer, uint mesh_id, MishMesh::TriMesh &mesh, const std::vector<OpenMesh::ArrayKernel::VertexHandle> vhs) {
	viewer.data(mesh_id).points.resize(0, 6);
	for(auto vh : vhs) {
		viewer.data(mesh_id).add_points(Eigen::RowVector3d(mesh.point(vh).data()), Eigen::RowVector3d{0, 0, 1});
	}
}

void visulalize_geodesics(MishMesh::TriMesh &mesh, MishMesh::TriMesh::VertexHandle vh, igl::opengl::glfw::Viewer &viewer, const MishMesh::GeodesicDistanceProperty distanceProperty) {
	MishMesh::compute_novotni_geodesics(mesh, vh, distanceProperty);
	Eigen::MatrixX3d colors(mesh.n_vertices(), 3);

	MishMesh::cosine_colorize_mesh(mesh, distanceProperty, 10);
	for(int i = 0; i < mesh.n_vertices(); i++) {
		colors.row(i) = Eigen::RowVector3d(OpenMesh::Vec3d(mesh.color(mesh.vertex_handle(i))).data());
	}
	colors /= 255.0;
	viewer.data(left_mesh).set_colors(colors);

	MishMesh::colorize_mesh(mesh, distanceProperty);
	for(int i = 0; i < mesh.n_vertices(); i++) {
		colors.row(i) = Eigen::RowVector3d(OpenMesh::Vec3d(mesh.color(mesh.vertex_handle(i))).data());
	}
	colors /= 255.0;
	viewer.data(right_mesh).set_colors(colors);

	set_points(viewer, left_mesh, mesh, {vh});
	set_points(viewer, right_mesh, mesh, {vh});
}

MishMesh::TriMesh::VertexHandle get_vh(const MishMesh::TriMesh &mesh, MishMesh::TriMesh::FaceHandle fh, Eigen::Vector3f bc) {
	auto vhs = MishMesh::face_vertices(mesh, fh);
	if(bc[0] >= 1 / 3.) {
		return vhs[0];
	} else if(bc[1] > 1 / 3.){
		return vhs[1];
	} else {
		return vhs[2];
	}
}

bool mouse_down(igl::opengl::glfw::Viewer &viewer, int button, int modifier, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, MishMesh::TriMesh &mesh, MishMesh::GeodesicDistanceProperty distanceProperty) {
	if(button != 1) return false;
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;

	int face_id;
	Eigen::Vector3f bc;
	if(igl::unproject_onto_mesh({x, y}, viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, face_id, bc)) {
		auto fh = mesh.face_handle(face_id);
		auto vh = get_vh(mesh, fh, bc);
		visulalize_geodesics(mesh, vh, viewer, distanceProperty);
	}
	return false;
}


void setup_viewer(igl::opengl::glfw::Viewer &viewer, uint mesh_id, Eigen::MatrixX3d V, Eigen::MatrixX3i F) {
	viewer.data(mesh_id).set_mesh(V, F);
	viewer.data(mesh_id).set_face_based(true);
	viewer.data(mesh_id).line_width = 1;
	viewer.data(mesh_id).point_size = 10;
	viewer.data(mesh_id).show_lines = false;
}

int main(int argc, char *argv[]) {
	MishMesh::TriMesh mesh;
	OpenMesh::IO::read_mesh(mesh, argv[1]);

	MishMesh::GeodesicDistanceProperty distanceProperty;
	mesh.add_property(distanceProperty);
	mesh.request_vertex_colors();

	Eigen::MatrixX3d V;
	Eigen::MatrixX3i F;
	std::tie(V, F) = MishMesh::convert_to_face_vertex_mesh(mesh);

	igl::opengl::glfw::Viewer viewer;

	left_mesh = viewer.data().id;
	setup_viewer(viewer, left_mesh, V, F);

	right_mesh = viewer.append_mesh();
	setup_viewer(viewer, right_mesh, V, F);

	// Split view viewer
	viewer.callback_init = [&](igl::opengl::glfw::Viewer &)
	{
		viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
		left_view = viewer.core_list[0].id;
		right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
		viewer.data(left_mesh).set_visible(false, right_view);
		viewer.data(right_mesh).set_visible(false, left_view);
		return false;
	};
	viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) {
		v.core(left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
		v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
		return true;
	};

	visulalize_geodesics(mesh, mesh.vertex_handle(0), viewer, distanceProperty);

	viewer.callback_mouse_down = [&V, &F, &mesh, &distanceProperty](igl::opengl::glfw::Viewer &viewer, int button, int modifier)->bool { return mouse_down(viewer, button, modifier, V, F, mesh, distanceProperty); };

	viewer.launch();
}
