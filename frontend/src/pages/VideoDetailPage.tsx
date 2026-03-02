import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getVideoVideosVideoIdGet, listJobsByVideoVideosVideoIdJobsGet } from "../api/generated";
import { authHeaders } from "../lib/auth";

export default function VideoDetailPage() {
  const navigate = useNavigate();
  const { id } = useParams();

  const { data, isLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => getVideoVideosVideoIdGet(id!, { headers: authHeaders() }),
  });

  const { data: jobsData } = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => listJobsByVideoVideosVideoIdJobsGet(id!, { headers: authHeaders() }),
  });

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">

      {/* 戻るボタン */}
      <button
        onClick={() => navigate("/")}
        className="mb-6 text-sm text-gray-500 hover:text-gray-700"
      >
        ← 一覧に戻る
      </button>

      {isLoading ? (
        <p>読み込み中...</p>
      ) : (
        <div className="space-y-6">

          {/* 動画情報カード */}
          {data?.status === 200 && (
            <div className="rounded-lg border bg-white p-6 shadow-sm">
              <div className="flex items-start justify-between">
                <h1 className="text-2xl font-bold text-gray-800">{data.data.title}</h1>
                <span className="rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-600">
                  {data.data.status}
                </span>
              </div>
              <p className="mt-2 text-sm text-gray-400">
                {new Date(data.data.created_at).toLocaleDateString("ja-JP")}
              </p>
            </div>
          )}

          {/* ジョブ一覧 */}
          {jobsData?.status === 200 && (
            <div>
              <h2 className="mb-3 text-lg font-semibold text-gray-700">処理ジョブ</h2>
              {jobsData.data.length === 0 ? (
                <p className="text-sm text-gray-400">ジョブはまだありません</p>
              ) : (
                <ul className="space-y-2">
                  {jobsData.data.map((job) => (
                    <li key={job.id} className="rounded-lg border bg-white p-4 shadow-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">ジョブ ID: {job.id.slice(0, 8)}...</span>
                        <span className="rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-600">
                          {job.status}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

        </div>
      )}
    </div>
  );
}
