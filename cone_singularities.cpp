#include <OpenMesh/Core/IO/MeshIO.hh>

#include <MishMesh/TriMesh.h>
#include <MishMesh/utils.h>
#include <MishMesh/cone_singularities.h>
#include <MishMesh/minimum_spanning_tree.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/unproject_onto_mesh.h>

void visualize_cones_and_mst(igl::opengl::glfw::Viewer &viewer, MishMesh::TriMesh &mesh, std::vector<MishMesh::TriMesh::VertexHandle> &cone_singularities) {
	viewer.data().points.resize(0, 3);
	viewer.data().lines.resize(0, 9);
	auto trees = MishMesh::minimum_spanning_trees(mesh, cone_singularities);

	for(auto vh : cone_singularities) {
		viewer.data().add_points(Eigen::RowVector3d(mesh.point(vh).data()), Eigen::RowVector3d{1, 0, 0});
	}

	std::vector<MishMesh::TriMesh::EdgeHandle> edges;
	for(const auto &tree : trees) {
		edges.insert(edges.end(), tree.begin(), tree.end());
	}
	Eigen::MatrixX3d P1(edges.size(), 3);
	Eigen::MatrixX3d P2(edges.size(), 3);
	for(uint i = 0; i < edges.size(); i++) {
		auto heh = mesh.halfedge_handle(edges[i], 0);
		P1.row(i) = Eigen::RowVector3d(mesh.point(mesh.from_vertex_handle(heh)).data());
		P2.row(i) = Eigen::RowVector3d(mesh.point(mesh.to_vertex_handle(heh)).data());
	}
	viewer.data().add_edges(P1, P2, Eigen::RowVector3d{0, 0, 1});
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

int main(int argc, char *argv[]) {
	MishMesh::TriMesh mesh;
	OpenMesh::IO::read_mesh(mesh, argv[1]);

	Eigen::MatrixX3d V;
	Eigen::MatrixX3i F;
	std::tie(V, F) = MishMesh::convert_to_face_vertex_mesh(mesh);

	MishMesh::cone_singularities::LaplaceSolver solver;
	MishMesh::cone_singularities::build_solver(solver, mesh, false);

	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().set_face_based(true);
	viewer.data().line_width = 2;
	viewer.data().point_size = 10;
	viewer.data().show_lines = false;

	MishMesh::ConeAdditionMode cone_addition_mode = MishMesh::ConeAdditionMode::MAX;
	int iterations = 5;
	int old_iterations = iterations;

	auto cone_singularities = MishMesh::cone_singularities::compute_cone_singularities(mesh, solver, 0.0, iterations, cone_addition_mode);
	visualize_cones_and_mst(viewer, mesh, cone_singularities);

	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	int cone_addition_mode_idx = 1;
	menu.callback_draw_viewer_menu = [&]() {
		if(ImGui::CollapsingHeader("Cone Singularities", ImGuiTreeNodeFlags_DefaultOpen)) {
			std::vector<std::string> choices{"BOTH", "MAX", "MIN"};
			if(ImGui::Combo("Cone Addition Mode", &cone_addition_mode_idx, choices)){
				switch(cone_addition_mode_idx) {
				case 0:
					cone_addition_mode = MishMesh::ConeAdditionMode::BOTH;
					break;
				case 1:
					cone_addition_mode = MishMesh::ConeAdditionMode::MAX;
					break;
				case 2:
					cone_addition_mode = MishMesh::ConeAdditionMode::MIN;
					break;
				}
				cone_singularities = MishMesh::cone_singularities::compute_cone_singularities(mesh, solver, 0.0, iterations, cone_addition_mode);
				visualize_cones_and_mst(viewer, mesh, cone_singularities);
			}

			if(ImGui::InputInt("Iterations", &iterations, 1, 3)){
				if(iterations < 0) {
					iterations = 0;
				}
				if(old_iterations > iterations) {
					if(cone_addition_mode == MishMesh::ConeAdditionMode::BOTH) {
						cone_singularities.resize(std::min(cone_singularities.size(), std::max(cone_singularities.size() - 2 * (old_iterations - iterations), static_cast<size_t>(0))));
					} else {
						cone_singularities.resize(std::min(cone_singularities.size(), std::max(cone_singularities.size() - 1 * (old_iterations - iterations), static_cast<size_t>(0))));
					}
				} else if(iterations > old_iterations) {
					std::set<size_t> cone_set;
					for(auto vh : cone_singularities) {
						cone_set.insert(vh.idx());
					}
					auto new_cone_singularities = MishMesh::cone_singularities::compute_cone_singularities(mesh, solver, 0.0, iterations - old_iterations, cone_addition_mode, cone_set);
					cone_singularities.insert(cone_singularities.end(), new_cone_singularities.begin(), new_cone_singularities.end());
				}
				old_iterations = iterations;
				visualize_cones_and_mst(viewer, mesh, cone_singularities);
			}
			if(ImGui::Button("Reset")) {
				cone_singularities = MishMesh::cone_singularities::compute_cone_singularities(mesh, solver, 0.0, iterations, cone_addition_mode);
				visualize_cones_and_mst(viewer, mesh, cone_singularities);
			}
		}
	};
	viewer.callback_mouse_down = [&V, &F, &mesh, &cone_singularities](igl::opengl::glfw::Viewer &viewer, int button, int modifier)->bool {
		if(button == 1) {
			double x = viewer.current_mouse_x;
			double y = viewer.core().viewport(3) - viewer.current_mouse_y;

			int face_id;
			Eigen::Vector3f bc;
			if(igl::unproject_onto_mesh({x, y}, viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, face_id, bc)) {
				auto fh = mesh.face_handle(face_id);
				auto vh = get_vh(mesh, fh, bc);
				cone_singularities.push_back(vh);
				visualize_cones_and_mst(viewer, mesh, cone_singularities);
				return true;
			}
		}
		return false;
	};

	viewer.launch();

	return 0;
}
