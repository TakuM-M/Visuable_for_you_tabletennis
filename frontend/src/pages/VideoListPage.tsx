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

  const statusColors = (status: string) => {
    switch (status) {
      case "processing":
        return "bg-yellow-100 text-yellow-600";
      case "completed":
        return "bg-green-100 text-green-600";
      case "failed":
        return "bg-red-100 text-red-600";
      default:
        return "bg-gray-100 text-gray-600";
    }
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">動画一覧</h1>
        <button
          onClick={() => navigate("/videos/new")}
          className="rounded bg-green-600 px-4 py-2 text-sm text-white hover:bg-green-700"
        >
          + 動画を追加
        </button>
      </div>

      {isLoading ? (
        <p>読み込み中...</p>
      ) : (
        <ul className="mt-4 space-y-3">
          {data?.data.map((video) => (
            <li
              key={video.id}
              onClick={() => navigate(`/videos/${video.id}`)}
              className="cursor-pointer rounded-lg border bg-white p-4 shadow-sm hover:shadow-md"
            >
              <div className="flex items-center justify-between">
                <p className="font-medium text-gray-800">{video.title}</p>
                <span className={`rounded-full px-2 py-1 text-xs ${statusColors(video.status)}`}>
                  {video.status}
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-400">
                {new Date(video.created_at).toLocaleDateString("ja-JP")}
              </p>
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
