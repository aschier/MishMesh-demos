#include <OpenMesh/Core/IO/MeshIO.hh>

#include <MishMesh/TriMesh.h>
#include <MishMesh/utils.h>
#include <MishMesh/dijkstra.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>

struct DijkstraPoints {
	MishMesh::TriMesh::VertexHandle source_vh;
	MishMesh::TriMesh::VertexHandle target_vh;
};

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

bool mouse_down(igl::opengl::glfw::Viewer &viewer, int button, int modifier, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, MishMesh::TriMesh &mesh, DijkstraPoints &dijkstraPoints) {
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;

	int face_id;
	Eigen::Vector3f bc;
	if(igl::unproject_onto_mesh({x, y}, viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, face_id, bc)) {
		auto fh = mesh.face_handle(face_id);
		auto vh = get_vh(mesh, fh, bc);
		if(button == 0) {
			dijkstraPoints.source_vh = vh;
		} else if(button == 2){
			dijkstraPoints.target_vh = vh;
		}
		viewer.data().points.resize(0, 3);
		viewer.data().lines.resize(0, 9);
		if(dijkstraPoints.source_vh.is_valid()) {
			Eigen::RowVector3d point(mesh.point(dijkstraPoints.source_vh).data());
			viewer.data().add_points(point, Eigen::RowVector3d(1, 0, 0));
		}
		if(dijkstraPoints.target_vh.is_valid()) {
			Eigen::RowVector3d point(mesh.point(dijkstraPoints.target_vh).data());
			viewer.data().add_points(point, Eigen::RowVector3d(0, 0, 1));
		}
		if(dijkstraPoints.source_vh.is_valid() && dijkstraPoints.target_vh.is_valid()) {
			auto result = MishMesh::dijkstra<MishMesh::TriMesh, MishMesh::L2HeuristicComparator<>>(dijkstraPoints.source_vh, dijkstraPoints.target_vh, mesh);
			Eigen::MatrixX3d colors(result.edges.size(), 3);
			Eigen::MatrixX3d P1(result.edges.size(), 3);
			Eigen::MatrixX3d P2(result.edges.size(), 3);
			for(int i = 0; i < result.edges.size(); i++){
				auto heh = mesh.halfedge_handle(result.edges[i], 0);
				P1.row(i) = Eigen::Vector3d(mesh.point(mesh.from_vertex_handle(heh)).data());
				P2.row(i) = Eigen::Vector3d(mesh.point(mesh.to_vertex_handle(heh)).data());
			}
			viewer.data().add_edges(P1, P2, Eigen::RowVector3d(0, 1, 0));
		}
	}
	return false;
}

int main(int argc, char *argv[]) {
	MishMesh::TriMesh mesh;
	OpenMesh::IO::read_mesh(mesh, argv[1]);

	Eigen::MatrixX3d V;
	Eigen::MatrixX3i F;
	std::tie(V, F) = MishMesh::convert_to_face_vertex_mesh(mesh);

	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().set_face_based(true);
	viewer.data().line_width = 2;
	viewer.data().point_size = 10;
	viewer.data().show_lines = false;

	DijkstraPoints dijkstraPoints;
	viewer.callback_mouse_down = [&V, &F, &mesh, &dijkstraPoints](igl::opengl::glfw::Viewer &viewer, int button, int modifier)->bool { return mouse_down(viewer, button, modifier, V, F, mesh, dijkstraPoints); };

	viewer.launch();
}
