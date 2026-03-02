import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { listVideosVideosGet } from "../api/generated";
import { authHeaders, removeToken } from "../lib/auth";

export default function VideoListPage() {
  const navigate = useNavigate();

  const {data, isLoading} = useQuery({
    queryKey: ["videos"],
    queryFn: () => listVideosVideosGet({ headers: authHeaders() }),
  });

  return (
    <div>
      <h1 className="text-2xl font-bold">動画一覧</h1>
      {isLoading ? (
        <p>読み込み中...</p>
      ) : (
        <ul>
          {data?.data.map((video) => (
            <li key={video.id} className="my-2">
              <button
                onClick={() => navigate(`/videos/${video.id}`)}
                className="text-blue-500 hover:underline"
              >
                {video.title}
              </button>
            </li>
          ))}
        </ul>
      )}
      <button
        onClick={() => {
          removeToken();
          navigate("/login");
        }}
        className="mt-4 rounded bg-red-500 px-4 py-2 text-white hover:bg-red-600"
      >
        ログアウト
      </button>
    </div>
  );
}